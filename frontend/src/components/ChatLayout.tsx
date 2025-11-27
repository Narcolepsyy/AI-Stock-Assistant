import React, { useCallback, useEffect, useState } from 'react'
import clsx from 'clsx'
import { Sidebar } from './layout'
import { ChatInput, ErrorDisplay, NetworkStatusIndicator, ChatMessagesArea, ChatModals } from './chat'
import { ErrorBoundary } from './common'
import WelcomeScreen from './WelcomeScreen'
import { translate } from '../i18n'
import { CITATION_STYLES } from '../constants'
import { generateId } from '../utils'
import type { Message, TabType, User } from '../types'
import { useNetworkStatus, useSettings, useChatSessionsContext, useChatContext } from '../hooks'
import { useUIState } from '../context/UIStateContext'

interface ChatLayoutProps {
  user: User
  onLogout: () => Promise<void> | void
}

const ChatLayout: React.FC<ChatLayoutProps> = ({ user, onLogout }) => {
  const { settings, setDeployment, updateSettings, models } = useSettings()
  const { uiState, updateUIState } = useUIState()
  const {
    sessions,
    currentSessionId,
    messages,
    createNewSession,
    switchToSession,
    deleteSession,
    addMessage,
    updateMessagesForSession,
    ensureSession,
  } = useChatSessionsContext()
  const {
    isLoading,
    isStreaming,
    error: chatError,
    activeTools,
    sendMessage,
    clearError,
    stopGeneration,
    conversationId,
  } = useChatContext()
  const network = useNetworkStatus()

  const [prompt, setPrompt] = useState('')
  const [pendingSessionId, setPendingSessionId] = useState<string | null>(null)

  const handleCreateNewSession = useCallback(() => {
    setPendingSessionId(null)
    createNewSession()
  }, [createNewSession])

  const handleSwitchToSession = useCallback((sessionId: string) => {
    setPendingSessionId(sessionId)
    switchToSession(sessionId)
  }, [switchToSession])

  const handleDeleteSession = useCallback((sessionId: string) => {
    if (pendingSessionId === sessionId) {
      setPendingSessionId(null)
    }
    deleteSession(sessionId)
  }, [deleteSession, pendingSessionId])

  useEffect(() => {
    const styleElement = document.createElement('style')
    styleElement.textContent = CITATION_STYLES
    document.head.appendChild(styleElement)

    return () => {
      document.head.removeChild(styleElement)
    }
  }, [])

  useEffect(() => {
    if (currentSessionId && pendingSessionId === currentSessionId) {
      setPendingSessionId(null)
    }
  }, [currentSessionId, pendingSessionId])

  const handleRewriteMessage = useCallback((message: Message) => {
    setPrompt(message.content)
  }, [])

  const handleCopyAnswer = useCallback(async (assistantMessage: Message) => {
    if (!assistantMessage?.content) return
    const text = assistantMessage.content
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
        return
      }
    } catch (err) {
      if (import.meta.env.DEV) {
        console.warn('Clipboard API failed, falling back to legacy copy.', err)
      }
    }

    try {
      const textarea = document.createElement('textarea')
      textarea.value = text
      textarea.style.position = 'fixed'
      textarea.style.opacity = '0'
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
    } catch (error) {
      if (import.meta.env.DEV) {
        console.error('Failed to copy answer', error)
      }
    }
  }, [])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault()
        updateUIState((prev) => ({ ...prev, showSearchModal: true }))
      } else if (event.key === 'Escape' && uiState.showSearchModal) {
        updateUIState((prev) => ({ ...prev, showSearchModal: false, searchModalQuery: '' }))
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [uiState.showSearchModal, updateUIState])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (uiState.showModelDropdown && !(event.target as Element)?.closest('[data-model-dropdown]')) {
        updateUIState((prev) => ({ ...prev, showModelDropdown: false }))
      }
      if (uiState.showUserMenu && !(event.target as Element)?.closest('[data-user-menu]')) {
        updateUIState((prev) => ({ ...prev, showUserMenu: false }))
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [uiState.showModelDropdown, uiState.showUserMenu, updateUIState])

  const handleChatSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prompt.trim() || isLoading || isStreaming) return

    clearError()
    const sessionId = ensureSession()
    setPendingSessionId(sessionId)

    const userMessage: Message = {
      id: generateId('user'),
      role: 'user',
      content: prompt,
      timestamp: new Date(),
    }

    // Add user message first
    addMessage(userMessage, sessionId)
    const currentPrompt = prompt
    setPrompt('')

    await sendMessage(
      {
        prompt: currentPrompt,
        system_prompt: settings.systemPrompt,
        deployment: settings.deployment,
        conversation_id: conversationId || undefined,
        locale: settings.locale,
        web_search_enabled: settings.webSearchEnabled,
      },
      // Pass updater that ensures user message is included
      (updater) => {
        updateMessagesForSession(sessionId, (prev) => {
          // Ensure user message is in the list before adding assistant message
          const hasUserMessage = prev.some(m => m.id === userMessage.id)
          const messagesWithUser = hasUserMessage ? prev : [...prev, userMessage]
          // Now apply the updater (which adds assistant message)
          return updater(messagesWithUser)
        })
      }
    )
  }, [prompt, isLoading, isStreaming, clearError, ensureSession, addMessage, sendMessage, settings, conversationId, updateMessagesForSession])

  const handleTabChange = useCallback((messageId: string, tab: TabType) => {
    updateUIState((prev) => ({
      ...prev,
      activeTab: { ...prev.activeTab, [messageId]: tab },
    }))
  }, [updateUIState])

  // Show the welcome screen only when we have no session AND no messages.
  // This prevents the first sent message from getting stuck on the blank welcome view
  // while the new session ID is still propagating through state updates.
  const hasActiveSession = Boolean(currentSessionId || pendingSessionId)
  const hasMessages = messages.length > 0
  const showCenteredWelcome = !hasActiveSession && !hasMessages && !isStreaming && !isLoading
  
  // Debug logging
  if (import.meta.env.DEV) {
    console.log('[ChatLayout] Render state:', {
      currentSessionId,
      pendingSessionId,
      messageCount: messages.length,
      hasActiveSession,
      hasMessages,
      showCenteredWelcome,
      isStreaming,
      isLoading,
    })
  }

  if (showCenteredWelcome) {
    return (
      <WelcomeScreen
        user={user}
        locale={settings.locale}
        models={models}
        currentDeployment={settings.deployment}
        prompt={prompt}
        onPromptChange={setPrompt}
        onSubmit={handleChatSubmit}
  onModelChange={setDeployment}
  isLoading={isLoading}
  isStreaming={isStreaming}
  error={chatError}
  onErrorDismiss={clearError}
  onStopGeneration={stopGeneration}
        networkStatus={network}
        chatSessions={sessions}
        uiState={uiState}
        onUIStateChange={updateUIState}
        settings={settings}
        onSettingsChange={updateSettings}
        onNewChat={handleCreateNewSession}
        onSessionSelect={handleSwitchToSession}
        onSessionDelete={handleDeleteSession}
        onLogout={onLogout}
      />
    )
  }

  return (
    <div className="relative h-screen bg-gray-900">
      <div
        className={clsx(
          'fixed left-0 top-0 h-full z-50 transition-all duration-300',
          uiState.isLeftSidebarExpanded ? 'w-80' : 'w-12'
        )}
      >
        <div className="h-full bg-gray-800/90 backdrop-blur-sm border-r border-gray-700 flex flex-col">
          <ErrorBoundary>
            <Sidebar
              isExpanded={uiState.isLeftSidebarExpanded}
              onToggleExpanded={() =>
                updateUIState((prev) => ({
                  ...prev,
                  isLeftSidebarExpanded: !prev.isLeftSidebarExpanded,
                }))
              }
              sessions={sessions}
              currentSessionId={currentSessionId}
              user={user}
              locale={settings.locale}
              onNewChat={handleCreateNewSession}
              onSearchClick={() =>
                updateUIState((prev) => ({ ...prev, showSearchModal: true }))
              }
              onSessionClick={handleSwitchToSession}
              onSessionDelete={handleDeleteSession}
              onSettingsClick={() =>
                updateUIState((prev) => ({ ...prev, showSettings: true, showUserMenu: false }))
              }
              onLogoutClick={() =>
                updateUIState((prev) => ({ ...prev, showLogoutConfirm: true, showUserMenu: false }))
              }
              showUserMenu={uiState.showUserMenu}
              onUserMenuToggle={() =>
                updateUIState((prev) => ({ ...prev, showUserMenu: !prev.showUserMenu }))
              }
            />
          </ErrorBoundary>
        </div>
      </div>

      <div className="absolute inset-0 flex flex-col bg-gray-900">
        <ErrorBoundary>
          <ChatMessagesArea
            messages={messages}
            isLoading={isLoading}
            isStreaming={isStreaming}
            activeTools={activeTools}
            activeTab={uiState.activeTab}
            locale={settings.locale}
            models={models}
            deployment={settings.deployment}
            onTabChange={handleTabChange}
            onRewriteMessage={handleRewriteMessage}
            onCopyAnswer={handleCopyAnswer}
          />
        </ErrorBoundary>

        <div className="mx-4 mb-2">
          <NetworkStatusIndicator isOnline={network.isOnline} connectionQuality={network.connectionQuality} />
        </div>

        {chatError && (
          <ErrorDisplay
            error={chatError}
            onRetry={handleChatSubmit}
            onDismiss={clearError}
            canRetry={!!prompt.trim() && !isLoading && !isStreaming}
          />
        )}

        <div className="sticky bottom-0 z-30 bg-gray-900">
          <ChatInput
            value={prompt}
            onChange={setPrompt}
            onSubmit={handleChatSubmit}
            onStop={stopGeneration}
            isLoading={isLoading}
            isStreaming={isStreaming}
            disabled={false}
            placeholder={translate('placeholder', settings.locale)}
            models={models || undefined}
            currentDeployment={settings.deployment}
            onModelChange={setDeployment}
          />
        </div>
      </div>

      <ChatModals
        showSettings={uiState.showSettings}
        onSettingsClose={() => updateUIState((prev) => ({ ...prev, showSettings: false }))}
        settings={settings}
        onSettingsChange={updateSettings}
        models={models}
        showLogoutConfirm={uiState.showLogoutConfirm}
        onLogoutClose={() => updateUIState((prev) => ({ ...prev, showLogoutConfirm: false }))}
        onLogoutConfirm={onLogout}
        showSearchModal={uiState.showSearchModal}
        onSearchClose={() =>
          updateUIState((prev) => ({ ...prev, showSearchModal: false, searchModalQuery: '' }))
        }
        searchModalQuery={uiState.searchModalQuery}
        onSearchQueryChange={(query: string) =>
          updateUIState((prev) => ({ ...prev, searchModalQuery: query }))
        }
        sessions={sessions}
        onSessionSelect={(sessionId: string) => {
          handleSwitchToSession(sessionId)
          updateUIState((prev) => ({ ...prev, showSearchModal: false, searchModalQuery: '' }))
        }}
        locale={settings.locale}
      />
    </div>
  )
}

export default ChatLayout
