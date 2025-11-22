/**
 * Main App component - refactored for better maintainability
 */

import React, { useState } from 'react'
import { AuthForm } from './components/auth'
import { LoadingSpinner } from './components/ui'

import ChatLayout from './components/ChatLayout'
import Dashboard from './components/Dashboard'
import StockDetail from './components/StockDetail'
import { useAuth, useSettings } from './hooks'
import { SettingsProvider } from './context/SettingsContext'
import { UIStateProvider } from './context/UIStateContext'
import { ChatProvider } from './context/ChatContext'
import { ChatSessionsProvider } from './context/ChatSessionsContext'

type View = 'chat' | 'dashboard' | 'stock-detail'

const AppContent: React.FC<{ auth: ReturnType<typeof useAuth> }> = ({ auth }) => {
  const { settings } = useSettings()
  const [currentView, setCurrentView] = useState<View>('chat')
  const [selectedStock, setSelectedStock] = useState<string>('')

  const handleStockSelect = (symbol: string) => {
    setSelectedStock(symbol)
    setCurrentView('stock-detail')
  }

  const handleBackToDashboard = () => {
    setCurrentView('dashboard')
    setSelectedStock('')
  }

  if (auth.isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <LoadingSpinner size="lg" label="Loading..." />
      </div>
    )
  }

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

  return (
    <ChatProvider>
      <ChatSessionsProvider>
        <div className="min-h-screen bg-gray-900">
          {/* Navigation Bar */}
          <nav className="bg-gray-800 border-b border-gray-700">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-16">
                <div className="flex items-center space-x-4">
                  <h1 className="text-xl font-bold text-white">AI Stocks Assistant</h1>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setCurrentView('chat')}
                      className={`px-4 py-2 rounded-lg font-medium transition-colors ${currentView === 'chat'
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-300 hover:bg-gray-700'
                        }`}
                    >
                      Chat
                    </button>
                    <button
                      onClick={() => setCurrentView('dashboard')}
                      className={`px-4 py-2 rounded-lg font-medium transition-colors ${currentView === 'dashboard'
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-300 hover:bg-gray-700'
                        }`}
                    >
                      Dashboard
                    </button>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  {/* Market Sentiment Indicator */}


                  <span className="text-gray-300 text-sm">{auth.user.username}</span>
                  <button
                    onClick={() => auth.logout()}
                    className="px-4 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    Logout
                  </button>
                </div>
              </div>
            </div>
          </nav>

          {/* View Content - Keep all mounted to preserve state */}
          <div className={currentView === 'chat' ? '' : 'hidden'}>
            <ChatLayout user={auth.user} onLogout={auth.logout} />
          </div>
          <div className={currentView === 'dashboard' ? '' : 'hidden'}>
            <Dashboard onStockSelect={handleStockSelect} />
          </div>
          <div className={currentView === 'stock-detail' ? '' : 'hidden'}>
            <StockDetail symbol={selectedStock} onBack={handleBackToDashboard} />
          </div>
        </div>
      </ChatSessionsProvider>
    </ChatProvider>
  )
}

const App: React.FC = () => {
  const auth = useAuth()

  return (
    <SettingsProvider>
      <UIStateProvider>
        <AppContent auth={auth} />
      </UIStateProvider>
    </SettingsProvider>
  )
}

export default App