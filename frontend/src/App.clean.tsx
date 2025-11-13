/**
 * Main App component - refactored for better maintainability
 */

import React, { useEffect, useState, useCallback } from 'react'
import { flushSync } from 'react-dom'
import { listModels, type ModelsResponse } from './lib/api'
import { useAuth, useChatSessions, useChat, useNetworkStatus } from './hooks'
import { AuthForm } from './components/auth'
import { ChatInput, ErrorDisplay, NetworkStatusIndicator } from './components/chat'
import { Sidebar } from './components/layout'
import { Button, Modal, Input, Textarea, LoadingSpinner } from './components/ui'
import { QASectionManager } from './components/QASectionManager'
import { QASection } from './components/QASection'
import WelcomeScreen from './components/WelcomeScreen'
import SearchModal from './components/SearchModal'
import { translate, getStoredLocale, setStoredLocale, type Locale } from './i18n'
import { STORAGE_KEYS, DEFAULT_SYSTEM_PROMPT, CITATION_STYLES } from './constants'
import { generateId } from './utils'
import type { Message, TabType, UIState, AppSettings } from './types'
import clsx from 'clsx'

// Error boundary for React error handling
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error boundary caught an error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="flex flex-col items-center justify-center min-h-[200px] p-4 bg-red-50 border border-red-200 rounded-lg">
          <h3 className="text-lg font-semibold text-red-800 mb-1">Something went wrong</h3>
          <p className="text-sm text-red-600 text-center mb-3">
            An error occurred while rendering this component
          </p>
          <Button
            onClick={() => this.setState({ hasError: false, error: undefined })}
            variant="danger"
            size="sm"
          >
            Try again
          </Button>
        </div>
      )
    }

    return this.props.children
  }
}

const App: React.FC = () => {
  // Auth state
  const auth = useAuth()

  // Settings state
  const [settings, setSettings] = useState<AppSettings>({
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    deployment: '',
    locale: getStoredLocale(),
  })

  // UI state
  const [uiState, setUIState] = useState<UIState>({
    showSettings: false,
    showModelDropdown: false,
    showSearchModal: false,
    searchModalQuery: '',
    showUserMenu: false,
    isLeftSidebarExpanded: false,
    showLogoutConfirm: false,
    activeTab: {},
  })

  // Models and chat state
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompt, setPrompt] = useState('')
  const [conversationId, setConversationId] = useState<string | null>(null)

  // Custom hooks
  const chatSessions = useChatSessions(settings.locale)
  const chat = useChat()
  const network = useNetworkStatus()

  // Inject citation styles on mount
  useEffect(() => {
    const styleElement = document.createElement('style')
    styleElement.textContent = CITATION_STYLES
    document.head.appendChild(styleElement)
    
    return () => {
      document.head.removeChild(styleElement)
    }
  }, [])

  // Load models on mount
  useEffect(() => {
    listModels()
      .then((m) => {
        setModels(m)
        try {
          const saved = localStorage.getItem(STORAGE_KEYS.SELECTED_MODEL)
          if (saved) {
            const found = m.available.find((x) => x.id === saved)
            setSettings(prev => ({ 
              ...prev, 
              deployment: found ? found.id : (m.default || '') 
            }))
          } else {
            setSettings(prev => ({ ...prev, deployment: m.default || '' }))
          }
        } catch {
          setSettings(prev => ({ ...prev, deployment: m.default || '' }))
        }
      })
      .catch((err) => {
        console.error('Failed to load models:', err)
        try {
          const saved = localStorage.getItem(STORAGE_KEYS.SELECTED_MODEL)
          if (saved) {
            setSettings(prev => ({ ...prev, deployment: saved }))
          }
        } catch {}
      })
  }, [])

  // Save model choice
  useEffect(() => {
    try {
      if (settings.deployment) {
        localStorage.setItem(STORAGE_KEYS.SELECTED_MODEL, settings.deployment)
      }
    } catch {}
  }, [settings.deployment])

  // Load system prompt from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEYS.SYSTEM_PROMPT)
      if (saved) {
        setSettings(prev => ({ ...prev, systemPrompt: saved }))
      }
    } catch {}
  }, [])

  // Save system prompt to localStorage
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEYS.SYSTEM_PROMPT, settings.systemPrompt)
    } catch {}
  }, [settings.systemPrompt])

  // Save locale preference
  useEffect(() => {
    setStoredLocale(settings.locale)
  }, [settings.locale])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault()
        setUIState(prev => ({ ...prev, showSearchModal: true }))
      } else if (event.key === 'Escape' && uiState.showSearchModal) {
        setUIState(prev => ({ 
          ...prev, 
          showSearchModal: false, 
          searchModalQuery: '' 
        }))
      }
    }
    
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [uiState.showSearchModal])

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (uiState.showModelDropdown && !(event.target as Element)?.closest('[data-model-dropdown]')) {
        setUIState(prev => ({ ...prev, showModelDropdown: false }))
      }
      if (uiState.showUserMenu && !(event.target as Element)?.closest('[data-user-menu]')) {
        setUIState(prev => ({ ...prev, showUserMenu: false }))
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [uiState.showModelDropdown, uiState.showUserMenu])

  // Chat handlers
  const handleChatSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim() || chat.isLoading || chat.isStreaming) return
    if (!auth.user) {
      chat.clearError()
      // Set error through chat state instead of separate error state
      return
    }

    const sessionId = chatSessions.ensureSession()
    
    // Add user message
    const userMessage: Message = {
      id: generateId('user'),
      role: 'user',
      content: prompt,
      timestamp: new Date(),
    }
    
    chatSessions.addMessage(userMessage)
    const currentPrompt = prompt
    setPrompt('')

    // Send message via chat hook
    await chat.sendMessage(
      {
        prompt: currentPrompt,
        system_prompt: settings.systemPrompt,
        deployment: settings.deployment,
        conversation_id: conversationId || undefined,
        locale: settings.locale,
      },
      chatSessions.updateLastMessage
    )

    // Update conversation ID if provided by chat hook
    if (chat.conversationId) {
      setConversationId(chat.conversationId)
    }
  }, [
    prompt, 
    chat, 
    auth.user, 
    chatSessions, 
    settings, 
    conversationId
  ])

  // Helper to get current active tab for a message
  const getCurrentTab = useCallback((messageId: string): TabType => {
    return uiState.activeTab[messageId] || 'answer'
  }, [uiState.activeTab])

  const handleTabChange = useCallback((messageId: string, tab: TabType) => {
    setUIState(prev => ({
      ...prev,
      activeTab: { ...prev.activeTab, [messageId]: tab }
    }))
  }, [])

  // Show loading screen during initial auth load
  if (auth.isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <LoadingSpinner size="lg" label="Loading..." />
      </div>
    )
  }

  // Show auth form if not authenticated
  if (!auth.user) {
    return (
      <AuthForm
        mode={auth.authMode}
        onModeChange={auth.setAuthMode}
        onSubmit={async (data) => {
          await (auth.authMode === 'login' ? auth.login(data) : auth.register(data))
        }}
        error={auth.error}
        isLoading={auth.isLoading}
        locale={settings.locale}
      />
    )
  }

  // Main chat interface
  const showCenteredWelcome = !chatSessions.currentSessionId

  if (showCenteredWelcome) {
    return (
      <WelcomeScreen
        user={auth.user}
        locale={settings.locale}
        models={models}
        currentDeployment={settings.deployment}
        prompt={prompt}
        onPromptChange={setPrompt}
        onSubmit={handleChatSubmit}
        onModelChange={(modelId: string) => setSettings(prev => ({ ...prev, deployment: modelId }))}
        isLoading={chat.isLoading}
        isStreaming={chat.isStreaming}
        error={chat.error}
        onErrorDismiss={chat.clearError}
        onStopGeneration={chat.stopGeneration}
        networkStatus={network}
        chatSessions={chatSessions.sessions}
        uiState={uiState}
        onUIStateChange={setUIState}
        settings={settings}
        onSettingsChange={setSettings}
        onNewChat={chatSessions.createNewSession}
        onSessionSelect={chatSessions.switchToSession}
        onSessionDelete={chatSessions.deleteSession}
        onLogout={auth.logout}
      />
    )
  }

  // Regular chat interface with sidebar and messages
  return (
    <div className="relative h-screen bg-gray-900">
      {/* Sidebar */}
      <div className={clsx(
        'fixed left-0 top-0 h-full z-50 transition-all duration-300',
        uiState.isLeftSidebarExpanded ? 'w-80' : 'w-12'
      )}>
        <div className="h-full bg-gray-800/90 backdrop-blur-sm border-r border-gray-700 flex flex-col">
          <ErrorBoundary>
            <Sidebar
              isExpanded={uiState.isLeftSidebarExpanded}
              onToggleExpanded={() => 
                setUIState(prev => ({ 
                  ...prev, 
                  isLeftSidebarExpanded: !prev.isLeftSidebarExpanded 
                }))
              }
              sessions={chatSessions.sessions}
              currentSessionId={chatSessions.currentSessionId}
              user={auth.user}
              locale={settings.locale}
              onNewChat={chatSessions.createNewSession}
              onSearchClick={() => 
                setUIState(prev => ({ ...prev, showSearchModal: true }))
              }
              onSessionClick={chatSessions.switchToSession}
              onSessionDelete={chatSessions.deleteSession}
              onSettingsClick={() => 
                setUIState(prev => ({ ...prev, showSettings: true, showUserMenu: false }))
              }
              onLogoutClick={() => 
                setUIState(prev => ({ 
                  ...prev, 
                  showLogoutConfirm: true, 
                  showUserMenu: false 
                }))
              }
              showUserMenu={uiState.showUserMenu}
              onUserMenuToggle={() =>
                setUIState(prev => ({ ...prev, showUserMenu: !prev.showUserMenu }))
              }
            />
          </ErrorBoundary>
        </div>
      </div>

      {/* Main Content */}
      <div className="absolute inset-0 flex flex-col bg-gray-900">
        {/* Messages Container */}
        <ErrorBoundary>
          <div className="flex-1 overflow-y-auto">
            <div className="p-4 space-y-4">
              {chatSessions.messages.length === 0 ? (
                <div className="flex-1 flex items-center justify-center">
                  <div className="text-center max-w-md">
                    <h3 className="text-xl font-semibold text-white mb-2">
                      {translate('onboardingTitle', settings.locale)}
                    </h3>
                    <p className="text-gray-400 mb-4">
                      {translate('onboardingBody', settings.locale)}
                    </p>
                    {models && settings.deployment && (
                      <div className="inline-flex items-center space-x-2 px-3 py-1 bg-blue-800/50 rounded-full text-sm text-blue-300">
                        <span>
                          {translate('usingModel', settings.locale)} {
                            models.available.find(m => m.id === settings.deployment)?.name || 'AI Model'
                          }
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <>
                  {chatSessions.messages.reduce((acc: React.ReactNode[], message, index) => {
                    if (message.role === 'user') {
                      const assistantMessage = chatSessions.messages[index + 1]?.role === 'assistant' 
                        ? chatSessions.messages[index + 1] 
                        : undefined
                      
                      const isCurrentlyStreaming = chat.isStreaming && 
                        assistantMessage && 
                        index === chatSessions.messages.length - 2
                      
                      acc.push(
                        <QASection
                          key={message.id}
                          userMessage={message}
                          assistantMessage={assistantMessage}
                          isActive={false}
                          activeTools={isCurrentlyStreaming ? chat.activeTools : []}
                          isStreaming={isCurrentlyStreaming}
                          currentTab={assistantMessage ? getCurrentTab(assistantMessage.id) : 'answer'}
                          onTabChange={(tab) => {
                            if (assistantMessage) {
                              handleTabChange(assistantMessage.id, tab)
                            }
                          }}
                        />
                      )
                    }
                    return acc
                  }, [])}

                  {/* Live Tool Activity Panel */}
                  {chat.activeTools.length > 0 && (
                    <div className="max-w-2xl mx-auto px-8 py-2">
                      <div className="bg-gray-800 border border-blue-600 rounded-lg px-4 py-3 shadow-sm">
                        <p className="text-xs font-medium text-blue-300 mb-2">
                          {translate('toolActivity', settings.locale)}
                        </p>
                        <div className="space-y-1">
                          {chat.activeTools.map((tool) => (
                            <div key={tool.name} className="flex items-center gap-2 text-xs">
                              <span className="text-gray-300">{tool.name}</span>
                              <span className={clsx(
                                'px-2 py-0.5 rounded text-xs',
                                tool.status === 'running' && 'bg-blue-100 text-blue-800',
                                tool.status === 'completed' && 'bg-green-100 text-green-800',
                                tool.status === 'error' && 'bg-red-100 text-red-800'
                              )}>
                                {tool.status}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </ErrorBoundary>

        {/* Network status indicator */}
        <div className="mx-4 mb-2">
          <NetworkStatusIndicator 
            isOnline={network.isOnline} 
            connectionQuality={network.connectionQuality}
          />
        </div>

        {/* Error display */}
        {chat.error && (
          <ErrorDisplay 
            error={chat.error} 
            onRetry={handleChatSubmit}
            onDismiss={chat.clearError}
            canRetry={!!prompt.trim() && !chat.isLoading && !chat.isStreaming}
          />
        )}

        {/* Chat Input */}
        <ChatInput
          value={prompt}
          onChange={setPrompt}
          onSubmit={handleChatSubmit}
          onStop={chat.stopGeneration}
          isLoading={chat.isLoading}
          isStreaming={chat.isStreaming}
          disabled={!chatSessions.currentSessionId}
          placeholder={chatSessions.currentSessionId 
            ? translate('placeholder', settings.locale) 
            : 'Create a new chat to start messaging...'
          }
          models={models || undefined}
          currentDeployment={settings.deployment}
          onModelChange={(modelId) => 
            setSettings(prev => ({ ...prev, deployment: modelId }))
          }
        />
      </div>

      {/* Settings Modal */}
      <Modal
        isOpen={uiState.showSettings}
        onClose={() => setUIState(prev => ({ ...prev, showSettings: false }))}
        title={translate('settings', settings.locale)}
      >
        <div className="space-y-6">
          <Textarea
            label={translate('systemPrompt', settings.locale)}
            value={settings.systemPrompt}
            onChange={(e) => setSettings(prev => ({ 
              ...prev, 
              systemPrompt: e.target.value 
            }))}
            rows={6}
            placeholder="Enter system prompt..."
          />

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              {translate('language', settings.locale)}
            </label>
            <select
              value={settings.locale}
              onChange={(e) => setSettings(prev => ({ 
                ...prev, 
                locale: e.target.value as Locale 
              }))}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="en">{translate('english', settings.locale)}</option>
              <option value="ja">{translate('japanese', settings.locale)}</option>
            </select>
          </div>

          {models && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                {translate('modelSelection', settings.locale)}
              </label>
              <select
                value={settings.deployment}
                onChange={(e) => setSettings(prev => ({ 
                  ...prev, 
                  deployment: e.target.value 
                }))}
                className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">{translate('selectModelPlaceholder', settings.locale)}</option>
                {models.available.map((model) => (
                  <option key={model.id} value={model.id} disabled={!model.available}>
                    {model.name} {model.default ? '(Default)' : ''} - {model.provider.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </Modal>

      {/* Logout Confirmation Modal */}
      <Modal
        isOpen={uiState.showLogoutConfirm}
        onClose={() => setUIState(prev => ({ ...prev, showLogoutConfirm: false }))}
        title={translate('signOut', settings.locale)}
        maxWidth="sm"
      >
        <p className="text-gray-300 mb-4">{translate('signOutConfirm', settings.locale)}</p>
        <div className="flex space-x-3">
          <Button
            onClick={() => setUIState(prev => ({ ...prev, showLogoutConfirm: false }))}
            variant="ghost"
            className="flex-1"
          >
            {translate('cancel', settings.locale)}
          </Button>
          <Button
            onClick={() => {
              setUIState(prev => ({ ...prev, showLogoutConfirm: false }))
              auth.logout()
            }}
            variant="danger"
            className="flex-1"
          >
            {translate('signOut', settings.locale)}
          </Button>
        </div>
      </Modal>

      {/* Search Modal */}
      <SearchModal
        isOpen={uiState.showSearchModal}
        onClose={() => setUIState(prev => ({ 
          ...prev, 
          showSearchModal: false, 
          searchModalQuery: '' 
        }))}
        query={uiState.searchModalQuery}
        onQueryChange={(query: string) => setUIState(prev => ({ 
          ...prev, 
          searchModalQuery: query 
        }))}
        sessions={chatSessions.sessions}
        onSessionSelect={(sessionId: string) => {
          chatSessions.switchToSession(sessionId)
          setUIState(prev => ({ 
            ...prev, 
            showSearchModal: false, 
            searchModalQuery: '' 
          }))
        }}
        locale={settings.locale}
      />
    </div>
  )
}

export default App