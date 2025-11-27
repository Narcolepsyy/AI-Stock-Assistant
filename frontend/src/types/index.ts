// Re-export types from API
import type { User, ModelsResponse } from '../lib/api'

// Core application types
export type { User, ModelsResponse }

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: ToolCall[]
  timestamp: Date
}

export interface ToolCall {
  id: string
  name: string
  result: unknown
}

export interface ChatSession {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
  messageCount: number
}

export interface LiveTool {
  name: string
  status: 'running' | 'completed' | 'error'
  error?: string
}

export type ConnectionQuality = 'good' | 'poor' | 'offline'

export type AuthMode = 'login' | 'register'

export type TabType = 'answer' | 'sources' | 'steps'

export interface ToolMeta {
  label: string
  icon: React.ComponentType<{ className?: string }>
  color: string
}

export interface AuthState {
  user: User | null
  isLoading: boolean
  error: string | null
}

export interface ChatState {
  messages: Message[]
  currentSessionId: string | null
  conversationId: string | null
  isLoading: boolean
  isStreaming: boolean
  isTyping: boolean
  error: string | null
  activeTools: LiveTool[]
  isCachedStream: boolean
}

export interface UIState {
  showSettings: boolean
  showModelDropdown: boolean
  showSearchModal: boolean
  searchModalQuery: string
  showUserMenu: boolean
  isLeftSidebarExpanded: boolean
  showLogoutConfirm: boolean
  activeTab: Record<string, TabType>
}

export interface AppSettings {
  systemPrompt: string
  deployment: string
  locale: 'en' | 'ja'
  webSearchEnabled: boolean
}

// Form event types
export interface ChatFormData {
  prompt: string
}

export interface AuthFormData {
  username: string
  password: string
  email?: string
}

// Component prop types
export interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
}

export interface ModelDropdownProps {
  models: ModelsResponse
  currentDeployment: string
  onModelChange: (modelId: string) => void
  onClose: () => void
}

export interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (e: React.FormEvent) => void
  onStop: () => void
  isLoading: boolean
  isStreaming: boolean
  disabled?: boolean
  placeholder?: string
  models?: ModelsResponse
  currentDeployment?: string
  onModelChange?: (modelId: string) => void
}

export interface NetworkStatusProps {
  isOnline: boolean
  connectionQuality: ConnectionQuality
}

export interface ErrorDisplayProps {
  error: string
  onRetry?: (e: React.FormEvent) => void
  onDismiss?: () => void
  canRetry?: boolean
}