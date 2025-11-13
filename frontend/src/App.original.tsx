import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import {flushSync} from 'react-dom'
import { chat as chatApi, chatStream, cancelChatStream, login as loginApi, register as registerApi, me as meApi, setToken, getToken, type User, logout as logoutApi, listModels, type ModelsResponse, apiBase, authHeaders } from './lib/api'
import { QASectionManager } from './components/QASectionManager'
import { QASection } from './components/QASection'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeRaw from 'rehype-raw'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import './katex-dark.css'

// Enhanced citation styles
const citationStyles = `
  .highlight-source {
    animation: highlight-pulse 2s ease-out;
    border-color: #3b82f6 !important;
    background-color: rgba(59, 130, 246, 0.1) !important;
  }
  
  @keyframes highlight-pulse {
    0% { 
      box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
      border-color: #3b82f6;
    }
    70% { 
      box-shadow: 0 0 0 10px rgba(59, 130, 246, 0);
      border-color: #3b82f6;
    }
    100% { 
      box-shadow: 0 0 0 0 rgba(59, 130, 246, 0);
    }
  }
  
  .citation-tooltip {
    animation: tooltip-fadein 0.2s ease-out;
  }
  
  @keyframes tooltip-fadein {
    from { opacity: 0; transform: translateX(-50%) translateY(-5px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
  }
  
  .citation-link {
    position: relative;
    z-index: 10;
  }
  
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  
  .source-item:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }
  
  .citation-tooltip {
    max-width: 320px;
    line-height: 1.4;
  }
`;

import {
  MessageCircle,
  Send,
  Plus,
  Settings,
  User as UserIcon,
  LogOut,
  Trash2,
  Bot,
  Menu,
  X,
  Loader2,
  ExternalLink,
  Calendar,
  TrendingUp,
  Cpu,
  Wrench,
  Check,
  XCircle,
  Square,
  ChevronDown
} from 'lucide-react'
import { Transition } from '@headlessui/react'
import clsx from 'clsx'
import { translate, getStoredLocale, setStoredLocale, type Locale } from './i18n'
// Additional icons for tool badges
import {
  FileText,
  DollarSign,
  Search as SearchIcon,
  LineChart,
  Newspaper,
  Shield,
  BookOpen,
  Globe,
  Database
} from 'lucide-react'

// Types
type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: Array<{ id: string; name: string; result: unknown }>
  timestamp: Date
}

// Add type for live tool activity
type LiveTool = { name: string; status: 'running' | 'completed' | 'error'; error?: string }

type ChatSession = {
  id: string
  title: string
  lastMessage: string
  timestamp: Date
  messageCount: number
}

const STORAGE_KEY = 'chat_sessions'
const CURRENT_SESSION_KEY = 'current_session'

// Helper: map tool names to friendly labels, icons, and color styles
const TOOL_META: Record<string, { label: string; icon: React.ComponentType<{ className?: string }>; color: string }> = {
  get_company_profile: { label: 'Company profile', icon: FileText, color: 'bg-purple-50 text-purple-700 border-purple-200' },
  get_stock_quote: { label: 'Live quote', icon: DollarSign, color: 'bg-green-50 text-green-700 border-green-200' },
  search_symbol: { label: 'Search symbol', icon: SearchIcon, color: 'bg-blue-50 text-blue-700 border-blue-200' },
  get_historical_prices: { label: 'Historical prices', icon: LineChart, color: 'bg-amber-50 text-amber-800 border-amber-200' },
  get_stock_news: { label: 'Stock news', icon: Newspaper, color: 'bg-cyan-50 text-cyan-700 border-cyan-200' },
  get_augmented_news: { label: 'Augmented news', icon: Newspaper, color: 'bg-cyan-50 text-cyan-700 border-cyan-200' },
  get_risk_assessment: { label: 'Risk assessment', icon: Shield, color: 'bg-rose-50 text-rose-700 border-rose-200' },
  rag_search: { label: 'Knowledge search', icon: BookOpen, color: 'bg-slate-50 text-slate-700 border-slate-200' },
  // Web search tools
  web_search: { label: 'Web search', icon: Globe, color: 'bg-emerald-50 text-emerald-700 border-emerald-200' },
  perplexity_search: { label: 'AI web search', icon: Globe, color: 'bg-indigo-50 text-indigo-700 border-indigo-200' },
  web_search_news: { label: 'News search', icon: Globe, color: 'bg-orange-50 text-orange-700 border-orange-200' },
  augmented_rag_search: { label: 'Enhanced search', icon: Globe, color: 'bg-violet-50 text-violet-700 border-violet-200' },
  financial_context_search: { label: 'Financial web search', icon: Globe, color: 'bg-teal-50 text-teal-700 border-teal-200' },
}

type ToolCall = NonNullable<Message['toolCalls']>[number]

function summarizeResult(res: unknown): string | null {
  try {
    if (res == null) return null
    const s = JSON.stringify(res)
    return s.length > 200 ? s.slice(0, 200) + '…' : s
  } catch {
    return String(res).slice(0, 200)
  }
}

// Network Status Indicator Component
const NetworkStatusIndicator = React.memo(function NetworkStatusIndicator({
  isOnline,
  connectionQuality
}: {
  isOnline: boolean;
  connectionQuality: 'good' | 'poor' | 'offline';
}) {
  // Only show for offline - ignore slow connection to reduce clutter
  if (isOnline) return null
  
  return (
    <div className="px-2 py-1 rounded border text-xs bg-red-900/50 border-red-600 text-red-300 flex items-center gap-1.5">
      <span>🔴</span>
      <span className="font-medium">Offline</span>
    </div>
  )
})

// Enhanced Error Display Component with retry functionality
const ErrorDisplay = React.memo(function ErrorDisplay({
  error,
  onRetry,
  onDismiss,
  canRetry = true
}: {
  error: string;
  onRetry?: (e: React.FormEvent) => void;
  onDismiss?: () => void;
  canRetry?: boolean;
}) {
  const isNetworkError = error.includes('🌐') || error.includes('Connection failed') || error.includes('Unable to reach server')
  const isTimeoutError = error.includes('⏱️') || error.includes('timed out')
  
  return (
    <div className="max-w-3xl mx-auto px-4 mb-2">
      <div className="bg-red-900/50 border border-red-600 rounded-xl p-2 text-sm text-red-300">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <span className="text-xs">❌</span>
            <span className="truncate text-xs">{error}</span>
          </div>
          
          <div className="flex gap-1 flex-shrink-0">
            {canRetry && onRetry && (
              <button
                onClick={onRetry}
                className="w-6 h-6 bg-blue-600 hover:bg-blue-700 text-white text-xs rounded flex items-center justify-center transition-colors"
                title="Retry request"
              >
                ↻
              </button>
            )}
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="w-6 h-6 bg-gray-600 hover:bg-gray-700 text-white text-xs rounded flex items-center justify-center transition-colors"
                title="Dismiss error"
              >
                ×
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
})

// Memoized message component for better performance
const MessageContent = React.memo(function MessageContent({ 
  content, 
  role 
}: { 
  content: string; 
  role: 'user' | 'assistant';
}) {
  const renderWithLinks = useCallback((text?: string): React.ReactNode => {
    if (!text) return null
    const rgx = /(https:\/\/[^\s<>()]+[^\s<>().,!?;:])/g
    const out: React.ReactNode[] = []
    let lastIdx = 0
    let m: RegExpExecArray | null
    while ((m = rgx.exec(text)) !== null) {
      const start = m.index
      const end = start + m[0].length
      if (start > lastIdx) out.push(text.slice(lastIdx, start))
      const url = m[0]
      out.push(
        <a key={`url-${start}-${end}`} href={url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-700 underline inline-flex items-center gap-1">
          {url}
          <ExternalLink className="w-3 h-3" />
        </a>
      )
      lastIdx = end
    }
    if (lastIdx < text.length) out.push(text.slice(lastIdx))
    return out
  }, [])

  if (role === 'user') {
    return (
      <span className="text-inherit">
        {renderWithLinks(content)}
      </span>
    )
  }

  return (
    <div className="prose prose-lg max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeRaw, rehypeKatex]}
        className="text-gray-100"
        components={{
          p: ({children}) => <p className="text-gray-100 leading-relaxed mb-4 last:mb-0">{children}</p>,
          ul: ({children}) => <ul className="text-gray-100 space-y-2 ml-6 list-disc">{children}</ul>,
          ol: ({children}) => <ol className="text-gray-100 space-y-2 ml-6 list-decimal">{children}</ol>,
          li: ({children}) => <li className="text-gray-100 leading-relaxed">{children}</li>,
          h1: ({children}) => <h1 className="text-gray-100 text-xl font-semibold mb-4">{children}</h1>,
          h2: ({children}) => <h2 className="text-gray-100 text-lg font-semibold mb-3">{children}</h2>,
          h3: ({children}) => <h3 className="text-gray-100 text-base font-semibold mb-2">{children}</h3>,
          code: ({children}) => <code className="bg-gray-800 text-green-400 px-1.5 py-0.5 rounded">{children}</code>,
          pre: ({children}) => <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto border border-gray-700">{children}</pre>,
          blockquote: ({children}) => <blockquote className="border-l-4 border-gray-600 pl-4 text-gray-300 italic">{children}</blockquote>,
          strong: ({children}) => <strong className="text-gray-100 font-semibold">{children}</strong>,
          em: ({children}) => <em className="text-gray-200 italic">{children}</em>
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
});


const ToolCallsPanel = React.memo(function ToolCallsPanel({ toolCalls }: { toolCalls: ToolCall[] }) {
  const [expanded, setExpanded] = React.useState<Record<string, boolean>>({})

  // Group by tool name with counts and keep the latest sample result
  const grouped = React.useMemo(() => {
    const m = new Map<string, { name: string; count: number; sample?: unknown }>()
    for (const t of toolCalls) {
      const key = t.name || 'unknown'
      const cur = m.get(key)
      if (cur) {
        m.set(key, { ...cur, count: cur.count + 1, sample: t.result ?? cur.sample })
      } else {
        m.set(key, { name: key, count: 1, sample: t.result })
      }
    }
    // Sort by name for stable order
    return Array.from(m.values()).sort((a, b) => a.name.localeCompare(b.name))
  }, [toolCalls])

  const toggleExpanded = useCallback((name: string) => {
    setExpanded((e) => ({ ...e, [name]: !e[name] }))
  }, [])

  if (!grouped.length) return null

  return (
    <div className="space-y-3">
      <p className="text-sm font-medium text-gray-300 mb-3">Data Sources</p>
      <div className="space-y-3">
        {grouped.map((g) => {
          const meta = TOOL_META[g.name]
          const Icon = meta?.icon || Wrench
          const label = meta?.label || g.name
          const color = meta?.color || 'bg-gray-50 text-gray-700 border-gray-200'
          const isOpen = !!expanded[g.name]
          return (
            <div key={g.name} className="bg-gray-800 border border-gray-700 rounded-lg p-3">
              <div className="flex items-center gap-3 mb-2">
                <span className={clsx("inline-flex items-center gap-2 px-3 py-1 rounded-full border text-sm", color)}>
                  <Icon className="w-4 h-4" />
                  <span className="font-medium">{label}</span>
                </span>
                <span className="text-gray-400 text-sm">× {g.count}</span>
                {g.sample !== undefined && (
                  <button
                    type="button"
                    className="ml-auto text-blue-400 hover:text-blue-300 text-sm"
                    onClick={() => toggleExpanded(g.name)}
                  >
                    {isOpen ? 'Hide details' : 'View details'}
                  </button>
                )}
              </div>
              {isOpen && (
                <div className="mt-2 bg-gray-900 border border-gray-600 rounded p-3 text-xs text-gray-300 whitespace-pre-wrap break-all max-h-40 overflow-y-auto">
                  {summarizeResult(g.sample) || 'No details available'}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
});

// Error Boundary component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="flex flex-col items-center justify-center min-h-[200px] p-4 bg-red-50 border border-red-200 rounded-lg">
          <XCircle className="w-8 h-8 text-red-500 mb-2" />
          <h3 className="text-lg font-semibold text-red-800 mb-1">Something went wrong</h3>
          <p className="text-sm text-red-600 text-center mb-3">
            An error occurred while rendering this component
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: undefined })}
            className="px-3 py-1 text-sm bg-red-100 hover:bg-red-200 text-red-800 rounded border border-red-300"
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Memoized session item component
const SessionItem = React.memo(function SessionItem({ 
  session, 
  isActive, 
  onClick, 
  onDelete 
}: {
  session: ChatSession;
  isActive: boolean;
  onClick: () => void;
  onDelete: (e: React.MouseEvent) => void;
}) {
  return (
    <div
      className={clsx(
        'group flex items-center justify-between p-2 rounded-lg cursor-pointer transition-colors',
        isActive
          ? 'bg-gray-700 text-white'
          : 'hover:bg-gray-700 text-gray-300'
      )}
      onClick={onClick}
    >
      <div className="flex items-center space-x-3 flex-1 min-w-0">
        <MessageCircle className="w-4 h-4 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium truncate">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ node, ...props }) => <span {...props} />,
                a: ({ node, ...props }) => <span {...props} />,
                h1: 'span', h2: 'span', h3: 'span', h4: 'span', h5: 'span', h6: 'span',
                ul: 'span', ol: 'span', li: 'span',
                blockquote: 'span', pre: 'span', table: 'span',
                thead: 'span', tbody: 'span', tr: 'span', td: 'span', th: 'span',
              }}
            >
              {session.title}
            </ReactMarkdown>
          </div>
        </div>
      </div>

      <button
        onClick={onDelete}
        className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-600 transition-all flex-shrink-0"
      >
        <Trash2 className="w-4 h-4" />
      </button>
    </div>
  )
});

export default function App() {
  // Auth state
  const [user, setUser] = useState<User | null>(null)
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login')
  const [authError, setAuthError] = useState<string | null>(null)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [isInitialLoading, setIsInitialLoading] = useState(true)

  // UI State
  const [showSettings, setShowSettings] = useState(false)
  const [showModelDropdown, setShowModelDropdown] = useState(false)
  const [showSearchModal, setShowSearchModal] = useState(false)
  const [searchModalQuery, setSearchModalQuery] = useState('')
  const [showUserMenu, setShowUserMenu] = useState(false)
  const [isLeftSidebarExpanded, setIsLeftSidebarExpanded] = useState(false)
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false)
  const [activeTab, setActiveTab] = useState<Record<string, 'answer' | 'sources' | 'steps'>>({})
  // Tab state for QASection components

  // Chat Sessions State
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [searchQuery, setSearchQuery] = useState('')

  // Chat State
  const [prompt, setPrompt] = useState('')
  const [systemPrompt, setSystemPrompt] = useState(`You are a helpful virtual assistant specializing in financial topics with access to both internal knowledge and real-time web search.

CRITICAL RULE: When you are uncertain about any information or don't have complete knowledge, ALWAYS use web_search or perplexity_search to find the most current and accurate information. Never guess or provide incomplete answers when you can search for better information.

Tool Usage Rules:

For LIVE/CURRENT financial data:
- Use get_stock_quote → current stock price or snapshot
- Use get_historical_prices → historical stock/index data
- Use get_company_profile → company details, financials, and fundamentals
- Use get_augmented_news → latest articles and headlines with enhanced context
- Use get_risk_assessment → real-time stock risk metrics

For CURRENT EVENTS, NEWS & UNKNOWN INFORMATION:
- ALWAYS use web_search or perplexity_search → for any recent events, breaking news, or information you're uncertain about
- Use web_search → for real-time information, current events, recent developments
- Use perplexity_search → for comprehensive analysis with AI synthesis and citations
- Use financial_context_search → for financial news and market analysis

For EDUCATIONAL/CONCEPTUAL topics:
- Use rag_search → financial concepts, ratios, indicators (P/E, RSI, etc.) from knowledge base
- If rag_search doesn't provide sufficient information, follow up with web_search for additional context
- Use augmented_rag_search → complex topics requiring both knowledge base and current information

For ANY UNCERTAIN OR INCOMPLETE KNOWLEDGE:
- NEVER provide partial or potentially outdated information
- ALWAYS use web_search or perplexity_search to get current, accurate information
- If you don't know something with confidence, search for it immediately
- Prefer recent, verified sources over assumptions

Always cite sources when using web search results. Provide clear, concise responses using the retrieved data. When information conflicts between sources, acknowledge this and present both perspectives. Format your answers naturally and conversationally.`)

  const [deployment, setDeployment] = useState('')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [chatLoading, setChatLoading] = useState(false)
  const [chatError, setChatError] = useState<string | null>(null)
  const [isTyping, setIsTyping] = useState(false)

  // Add streaming state variables
  const [isStreaming, setIsStreaming] = useState(false)

  // Live tool activity state
  const [activeTools, setActiveTools] = useState<LiveTool[]>([])
  const [isCachedStream, setIsCachedStream] = useState(false)

  // Network connection monitoring
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [connectionQuality, setConnectionQuality] = useState<'good' | 'poor' | 'offline'>('good')

  // Models state
  const [models, setModels] = useState<ModelsResponse | null>(null)
  // Locale
  const [locale, setLocale] = useState<Locale>(getStoredLocale())

  // Refs
  const endRef = useRef<HTMLDivElement | null>(null)
  const messagesContainerRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // Chat Session Management
  const loadChatSessions = useCallback(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const sessions = JSON.parse(stored).map((s: any) => ({
          ...s,
          timestamp: new Date(s.timestamp)
        }))
        setChatSessions(sessions.sort((a: ChatSession, b: ChatSession) => b.timestamp.getTime() - a.timestamp.getTime()))
      }
    } catch (error) {
      console.error('Failed to load chat sessions:', error)
    }
  }, [])

  const saveChatSessions = useCallback((sessions: ChatSession[]) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions))
      setChatSessions(sessions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()))
    } catch (error) {
      console.error('Failed to save chat sessions:', error)
    }
  }, [])

  const loadCurrentSession = useCallback(() => {
    try {
      const sessionId = localStorage.getItem(CURRENT_SESSION_KEY)
      if (sessionId) {
        setCurrentSessionId(sessionId)
        const sessionMessages = localStorage.getItem(`messages_${sessionId}`)
        if (sessionMessages) {
          const parsed = JSON.parse(sessionMessages).map((m: any) => ({
            ...m,
            timestamp: new Date(m.timestamp)
          }))
          setMessages(parsed)
        } else {
          setMessages([])
        }
      }
    } catch (error) {
      console.error('Failed to load current session:', error)
    }
  }, [])

  const saveCurrentSession = useCallback((sessionId: string, sessionMessages: Message[]) => {
    try {
      localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
      localStorage.setItem(`messages_${sessionId}`, JSON.stringify(sessionMessages))
    } catch (error) {
      console.error('Failed to save current session:', error)
    }
  }, [])

  // Initialize app
  useEffect(() => {
    // Inject citation styles
    const styleElement = document.createElement('style');
    styleElement.textContent = citationStyles;
    document.head.appendChild(styleElement);
    
    const tok = getToken()
    if (!tok) {
      setIsInitialLoading(false)
      return
    }

    meApi().then(user => {
      setUser(user)
      loadChatSessions()
      loadCurrentSession()
      setIsInitialLoading(false)

      // Auto-activate existing session if no current session is set
      setTimeout(() => {
        const stored = localStorage.getItem(STORAGE_KEY)
        const currentSession = localStorage.getItem(CURRENT_SESSION_KEY)
        
        // Parse existing sessions to check if we have any
        let existingSessions: ChatSession[] = []
        try {
          if (stored) {
            existingSessions = JSON.parse(stored).map((s: any) => ({
              ...s,
              timestamp: new Date(s.timestamp)
            }))
          }
        } catch (error) {
          console.error('Failed to parse stored sessions:', error)
        }

        // Only process if we don't already have a current session
        if (!currentSession) {
          // If sessions exist but no current session is set, activate the most recent one
          if (existingSessions.length > 0) {
            console.log('Found existing sessions, activating most recent one')
            // Sort sessions by timestamp to get the most recent one
            const sortedSessions = existingSessions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
            const mostRecentSession = sortedSessions[0]
            
            if (mostRecentSession) {
              try {
                localStorage.setItem(CURRENT_SESSION_KEY, mostRecentSession.id)
                setCurrentSessionId(mostRecentSession.id)
                
                // Load messages for this session
                const sessionMessages = localStorage.getItem(`messages_${mostRecentSession.id}`)
                if (sessionMessages) {
                  const parsed = JSON.parse(sessionMessages).map((m: any) => ({
                    ...m,
                    timestamp: new Date(m.timestamp)
                  }))
                  setMessages(parsed)
                } else {
                  setMessages([])
                }
              } catch (error) {
                console.error('Failed to activate existing session:', error)
              }
            }
          } else {
            console.log('No existing sessions and no current session - ready for welcome screen')
          }
        }
      }, 100)

    }).catch(() => {
      setToken(null)
      setIsInitialLoading(false)
    })
  }, [loadChatSessions, loadCurrentSession, locale])

  // Load models with enhanced model selector
  useEffect(() => {
    listModels().then((m) => {
      setModels(m)
      console.log('Loaded models:', m)
      try {
        const saved = localStorage.getItem('selected_model')
        if (saved) {
          const found = m.available.find((x) => x.id === saved)
          setDeployment(found ? found.id : (m.default || ''))
        } else {
          // Use default model if no saved preference
          setDeployment(m.default || '')
        }
      } catch {
        setDeployment(m.default || '')
      }
    }).catch((err) => {
      console.error('Failed to load models:', err)
      try {
        const saved = localStorage.getItem('selected_model')
        if (saved) setDeployment(saved)
      } catch {}
    })
  }, [])

  // Save model choice
  useEffect(() => {
    try {
      if (deployment) localStorage.setItem('selected_model', deployment)
    } catch {}
  }, [deployment])

  // Memoized auto-scroll handler
  const scrollToBottom = useCallback(() => {
    const el = messagesContainerRef.current
    if (!el) return
    // Scroll smoothly to bottom without affecting page scroll
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [])

  // Helper to get current active tab for a message (unified state)
  const getCurrentTab = useCallback((messageId: string): 'answer' | 'sources' | 'steps' => {
    return activeTab[messageId] || 'answer'
  }, [activeTab])



  // Auto-scroll to bottom (confined to messages container)
  useEffect(() => {
    // Small delay to prevent conflicts with user scrolling
    const timeoutId = setTimeout(() => {
      scrollToBottom()
    }, 100)
    
    return () => clearTimeout(timeoutId)
  }, [messages.length, chatLoading, isTyping, scrollToBottom])

  // Helper function to determine if an AI response is long
  const isLongResponse = useCallback((messageIndex: number): boolean => {
    if (messageIndex < 0 || messageIndex >= messages.length) return false
    const message = messages[messageIndex]
    if (!message || message.role !== 'assistant') return false
    
    // Consider a response long if it has more than 300 characters or multiple paragraphs
    const content = message.content || ''
    const hasLongContent = content.length > 300
    const hasMultipleParagraphs = content.includes('\n\n') || content.split('\n').length > 3
    const hasToolCalls = Boolean(message.toolCalls && message.toolCalls.length > 0)
    
    return hasLongContent || hasMultipleParagraphs || hasToolCalls
  }, [messages])





  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [prompt])

  // Network connection monitoring
  useEffect(() => {
    let connectionCheckInterval: number | null = null
    let lastCheckTime = Date.now()
    
    // Monitor online/offline events
    const handleOnline = () => {
      setIsOnline(true)
      setConnectionQuality('good')
    }
    
    const handleOffline = () => {
      setIsOnline(false)
      setConnectionQuality('offline')
    }
    
    // Periodic connection quality check
    const checkConnectionQuality = async () => {
      if (!isOnline) return
      
      try {
        const start = Date.now()
        // Use the chat/models endpoint which is lighter than health check
        // Create timeout controller for compatibility
        const timeoutController = new AbortController()
        const timeoutId = setTimeout(() => timeoutController.abort(), 5000)
        
        const response = await fetch(`${apiBase()}/chat/models`, { 
          method: 'HEAD',
          cache: 'no-cache',
          headers: authHeaders(),
          signal: timeoutController.signal
        })
        
        clearTimeout(timeoutId)
        const responseTime = Date.now() - start
        
        if (response.ok || response.status === 401) { // 401 is still a server response
          setConnectionQuality(responseTime > 3000 ? 'poor' : 'good')
        } else {
          setConnectionQuality('poor')
        }
      } catch (error) {
        console.warn('Connection quality check failed:', error)
        setConnectionQuality('poor')
      }
    }
    
    // Set up event listeners
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    
    // Initial connection state
    setIsOnline(navigator.onLine)
    
    // Start periodic checks every 30 seconds when online
    if (navigator.onLine) {
      checkConnectionQuality() // Initial check
      connectionCheckInterval = window.setInterval(checkConnectionQuality, 30000)
    }
    
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
      if (connectionCheckInterval) {
        window.clearInterval(connectionCheckInterval)
      }
    }
  }, [isOnline])



  // Close model dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showModelDropdown && !(event.target as Element)?.closest('[data-model-dropdown]')) {
        setShowModelDropdown(false)
      }
      if (showUserMenu && !(event.target as Element)?.closest('[data-user-menu]')) {
        setShowUserMenu(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showModelDropdown, showUserMenu])

  // Keyboard shortcuts for search (Cmd/Ctrl + K to open, Escape to close)
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
        event.preventDefault()
        setShowSearchModal(true)
      } else if (event.key === 'Escape' && showSearchModal) {
        setShowSearchModal(false)
        setSearchModalQuery('')
      }
    }
    
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [showSearchModal])

  // Debug session state changes
  useEffect(() => {
    console.log('Session state changed:', {
      currentSessionId,
      messagesLength: messages.length,
      showCenteredWelcome: !currentSessionId
    })
  }, [currentSessionId, messages.length])

  const createNewSession = useCallback(() => {
    // Clear current session to show welcome screen
    setCurrentSessionId(null)
    setMessages([])
    setConversationId(null)
    setChatError(null)
    
    // Clear current session from localStorage
    try {
      localStorage.removeItem(CURRENT_SESSION_KEY)
    } catch (error) {
      console.error('Failed to clear current session from localStorage:', error)
    }
    
    console.log('Cleared current session - welcome screen will be shown')
  }, [locale])

  const switchToSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId)
    
    try {
      const sessionMessages = localStorage.getItem(`messages_${sessionId}`)
      if (sessionMessages) {
        const parsed = JSON.parse(sessionMessages).map((m: any) => ({
          ...m,
          timestamp: new Date(m.timestamp)
        }))
        setMessages(parsed)
      } else {
        setMessages([])
      }
      localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
    } catch (error) {
      console.error('Failed to switch session:', error)
      setMessages([])
    }
  }, [])

  // Helper function to strip markdown formatting from text
  const stripMarkdown = useCallback((text: string): string => {
    return text
      .replace(/\*\*([^*]+)\*\*/g, '$1')  // Remove **bold**
      .replace(/\*([^*]+)\*/g, '$1')      // Remove *italic*
      .replace(/`([^`]+)`/g, '$1')        // Remove `code`
      .replace(/~~([^~]+)~~/g, '$1')      // Remove ~~strikethrough~~
      .replace(/#{1,6}\s+/g, '')          // Remove headers (#, ##, etc.)
      .replace(/^\s*[-*+]\s+/gm, '')      // Remove list markers
      .replace(/^\s*\d+\.\s+/gm, '')      // Remove numbered list markers
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links [text](url)
      .trim()
  }, [])

  const updateSessionAfterMessage = useCallback((sessionId: string, newMessage: Message) => {
    setChatSessions(prevSessions => {
      const updatedSessions = prevSessions.map(session => {
        if (session.id === sessionId) {
          // Strip markdown from content before creating title
          const cleanContent = stripMarkdown(newMessage.content)
          const title = newMessage.role === 'user'
            ? cleanContent.slice(0, 30) + (cleanContent.length > 30 ? '...' : '')
            : session.title !== translate('newChat', locale) ? session.title : cleanContent.slice(0, 30) + (cleanContent.length > 30 ? '...' : '')

          return {
            ...session,
            title,
            lastMessage: newMessage.content.slice(0, 100) + (newMessage.content.length > 100 ? '...' : ''),
            timestamp: new Date(),
            messageCount: session.messageCount + 1
          }
        }
        return session
      })

      // Save to localStorage
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedSessions))
      } catch (error) {
        console.error('Failed to save chat sessions:', error)
      }
      
      return updatedSessions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
    })
  }, [locale, stripMarkdown])

  const deleteSession = useCallback((sessionId: string) => {
    const updatedSessions = chatSessions.filter(s => s.id !== sessionId)
    saveChatSessions(updatedSessions)

    try {
      localStorage.removeItem(`messages_${sessionId}`)
    } catch (error) {
      console.error('Failed to delete session messages:', error)
    }

    if (currentSessionId === sessionId) {
      if (updatedSessions.length > 0 && updatedSessions[0]) {
        switchToSession(updatedSessions[0].id)
      } else {
        // Don't auto-create a new session, just clear the current session
        setCurrentSessionId(null)
        setMessages([])
        setConversationId(null)
        setChatError(null)
        try {
          localStorage.removeItem(CURRENT_SESSION_KEY)
        } catch (error) {
          console.error('Failed to clear current session from localStorage:', error)
        }
      }
    }
  }, [chatSessions, currentSessionId, saveChatSessions, switchToSession])

  // Memoized filtered sessions with optimized search
  const filteredSessions = useMemo(() => {
    if (!searchQuery.trim()) {
      return chatSessions
    }

    const query = searchQuery.toLowerCase().trim()
    return chatSessions.filter(session => 
      session.title.toLowerCase().includes(query) || 
      session.lastMessage.toLowerCase().includes(query)
    )
  }, [chatSessions, searchQuery])

  // Chat handlers
  const onSubmitChat = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    // Prevent duplicate sends while loading/streaming
  if (!prompt.trim() || chatLoading || isStreaming) return
  if (!user) { setChatError(translate('mustSignIn', locale)); return }

    // Close model dropdown if open
    setShowModelDropdown(false)
    
    setChatError(null)
    setChatLoading(true)
    setActiveTools([]) // reset tool activity at start of a request

    // Auto-create session if we don't have one (welcome screen case)
    let sessionId = currentSessionId
    if (!sessionId) {
      console.log('Creating new session from welcome screen...')
      
      // Create a new session for the first message
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      const newSession: ChatSession = {
        id: newSessionId,
        title: translate('newChat', locale),
        lastMessage: '',
        timestamp: new Date(),
        messageCount: 0
      }

      // Update sessions using functional state update to avoid stale closure
      setChatSessions(prevSessions => {
        const updatedSessions = [newSession, ...prevSessions]
        // Save to localStorage
        try {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedSessions))
        } catch (error) {
          console.error('Failed to save chat sessions:', error)
        }
        return updatedSessions.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      })
      
      // Set session ID immediately and synchronously with flushSync
      sessionId = newSessionId
      flushSync(() => {
        setCurrentSessionId(newSessionId)
      })
      
      // Save the new session as current
      saveCurrentSession(newSessionId, [])
      
      console.log('New session created:', newSessionId)
      
      // Force a small delay to ensure state propagation
      await new Promise(resolve => setTimeout(resolve, 50))
    }

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: prompt,
      timestamp: new Date()
    }

    // Use functional update to get current messages and avoid stale closure
    setMessages(prevMessages => {
      const newMessages = [...prevMessages, userMsg]
      updateSessionAfterMessage(sessionId, userMsg)
      saveCurrentSession(sessionId, newMessages)
      return newMessages
    })

    const currentPrompt = prompt
    setPrompt('')

    // Use streaming API with improved SSE handling
  setIsStreaming(true)
  setIsTyping(true)

    // Add placeholder message for streaming content
    const streamingMsg: Message = {
      id: `a-${Date.now()}`,
      role: 'assistant',
      content: '',
      timestamp: new Date()
    }
    
    // Add streaming message using functional update
    setMessages(prevMessages => [...prevMessages, streamingMsg])

    // Use a ref to track accumulated content to avoid closure issues
    let accumulatedContent = ''

    try {
      await chatStream(
        {
          prompt: currentPrompt,
          system_prompt: systemPrompt,
          deployment,
          conversation_id: conversationId || undefined,
          locale,
        },
        // onChunk - handle real-time content
        (chunk) => {
          if (chunk.type === 'start') {
            if (chunk.conversation_id) setConversationId(chunk.conversation_id)
            if (chunk.cached === true) setIsCachedStream(true)
            console.log('SSE stream started:', chunk.model)
          } else if (chunk.type === 'content' && chunk.delta) {
            // Accumulate content
            accumulatedContent += chunk.delta

            // Update the streaming message in real-time (immutably)
            setMessages(prevMessages => {
              if (prevMessages.length === 0) return prevMessages
              const lastIndex = prevMessages.length - 1
              const lastMsg = prevMessages[lastIndex]
              if (!lastMsg || lastMsg.role !== 'assistant') return prevMessages
              const updatedLast: Message = { ...lastMsg, content: accumulatedContent }
              return [...prevMessages.slice(0, lastIndex), updatedLast]
            })
          } else if (chunk.type === 'tool_calls') {
            // Show tool execution indicator and add tool entries
            const names: string[] = Array.isArray(chunk.tools) ? chunk.tools : []
            setActiveTools(prev => {
              const existing = new Map(prev.map(t => [t.name, t]))
              names.forEach(n => {
                if (!existing.has(n)) existing.set(n, { name: n, status: 'running' })
              })
              return Array.from(existing.values())
            })
            setIsTyping(true) // Keep typing indicator during tool execution
          } else if (chunk.type === 'tool_call') {
            // Individual tool updated/completed; insert if missing
            const name = chunk.name as string | undefined
            const status = (chunk.status as 'completed' | 'error' | 'running' | undefined) || 'running'
            setActiveTools(prev => {
              const exists = prev.some(t => t.name === name)
              if (exists) return prev.map(t => t.name === name ? { ...t, status, error: chunk.error } : t)
              return [...prev, { name: name || 'unknown', status, error: chunk.error }]
            })
          }
        },
        // onComplete - finalize the response
        (data) => {
          const finalAssistantMsg: Message = {
            ...streamingMsg,
            content: accumulatedContent,
            toolCalls: data.tool_calls || [],
            timestamp: new Date()
          }

          // Update messages using functional update to get latest state
          setMessages(prevMessages => {
            // Replace the streaming placeholder with final message
            const finalMessages = prevMessages.slice(0, -1).concat(finalAssistantMsg)
            updateSessionAfterMessage(sessionId, finalAssistantMsg)
            saveCurrentSession(sessionId, finalMessages)
            return finalMessages
          })

          // Update conversation ID if provided
          if (data.conversation_id) {
            setConversationId(data.conversation_id)
          }

          // Reset streaming state
          setIsStreaming(false)
          setIsTyping(false)
          setChatLoading(false)
          // Keep tool panel visible briefly after completion
          setTimeout(() => setActiveTools([]), 2000)
          setIsCachedStream(false)
        },
        // onError - handle errors
        (error) => {
          setChatError(error)
          setIsStreaming(false)
          setIsTyping(false)
          setChatLoading(false)
          // Keep any error states for a bit then clear
          setTimeout(() => setActiveTools([]), 3000)
          setIsCachedStream(false)
        }
      )
    } catch (err: any) {
      console.error('Chat stream error:', err)
      const errorMessage = err?.message || translate('streamingFailed', locale)
      
      // Remove the placeholder streaming message if it exists
      setMessages(prevMessages => {
        if (prevMessages.length > 0 && 
            prevMessages[prevMessages.length - 1]?.role === 'assistant' && 
            prevMessages[prevMessages.length - 1]?.content === '') {
          return prevMessages.slice(0, -1)
        }
        return prevMessages
      })
      
      setChatError(errorMessage)
      setIsStreaming(false)
      setIsTyping(false)
      setChatLoading(false)
      setTimeout(() => setActiveTools([]), 3000)
      setIsCachedStream(false)
    }
  }, [prompt, chatLoading, isStreaming, user, locale, currentSessionId, systemPrompt, deployment, conversationId, saveCurrentSession, updateSessionAfterMessage])

  const onLogout = async () => {
    try { await logoutApi() } catch {}
    setToken(null)
    setUser(null)
    setChatSessions([])
    setMessages([])
    setCurrentSessionId(null)
    try {
      localStorage.removeItem(STORAGE_KEY)
      localStorage.removeItem(CURRENT_SESSION_KEY)
    } catch {}
  }

  const onComposerKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      const form = (e.currentTarget as HTMLTextAreaElement).closest('form') as HTMLFormElement | null
      form?.requestSubmit()
    }
  }

  const onStopGenerating = useCallback(() => {
    const didCancel = cancelChatStream()
    // Regardless of cancel result, bring UI back to idle
    setIsStreaming(false)
    setIsTyping(false)
    setChatLoading(false)
    // Keep the placeholder bubble as-is; it will remain with whatever content streamed so far
    // Hide any tool activity indicators shortly
    setTimeout(() => setActiveTools([]), 1000)
  }, [])

  // Auth handlers
  const submitLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthError(null)
    try {
      const res = await loginApi(username, password)
      setToken(res.access_token)
      setUser(res.user)
      loadChatSessions()

    } catch (err: any) {
      setAuthError(err?.message || translate('loginFailed', locale))
    }
  }

  const submitRegister = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthError(null)
    try {
      await registerApi({ username, password, email: email || undefined })
      const res = await loginApi(username, password)
      setToken(res.access_token)
      setUser(res.user)
      loadChatSessions()
    } catch (err: any) {
      setAuthError(err?.message || translate('registerFailed', locale))
    }
  }



  // Show loading screen during initial app load
  if (isInitialLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
            <TrendingUp className="w-8 h-8 text-blue-600" />
          </div>
          <div className="flex items-center space-x-2 text-white">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Loading...</span>
          </div>
        </div>
      </div>
    )
  }

  // If not authenticated, show auth form
  if (!user) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4">
        <div className="max-w-md w-full bg-gray-800 rounded-xl shadow-lg p-8 border border-gray-700">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
              <TrendingUp className="w-8 h-8 text-blue-600" />
            </div>
            <h1 className="text-2xl font-bold text-white">{translate('appTitle', locale)}</h1>
            <p className="text-gray-400 mt-2">{translate('appSubtitle', locale)}</p>
          </div>

          <div className="flex mb-6 bg-gray-700 rounded-lg p-1">
            <button
              type="button"
              onClick={() => setAuthMode('login')}
              className={clsx(
                'flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors',
                authMode === 'login'
                  ? 'bg-gray-600 text-white shadow-sm'
                  : 'text-gray-300 hover:text-white'
              )}
              >
              {translate('authLogin', locale)}
            </button>
            <button
              type="button"
              onClick={() => setAuthMode('register')}
              className={clsx(
                'flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors',
                authMode === 'register'
                  ? 'bg-gray-600 text-white shadow-sm'
                  : 'text-gray-300 hover:text-white'
              )}
              >
              {translate('authRegister', locale)}
            </button>
          </div>

          <form onSubmit={authMode === 'login' ? submitLogin : submitRegister}>
            <div className="space-y-4">
              <div>
                <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-1">
                  Username
                </label>
                <input
                  id="username"
                  name="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-700 text-white"
                  required
                />
              </div>

              {authMode === 'register' && (
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-1">
                    Email (optional)
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-700 text-white"
                  />
                </div>
              )}

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-1">
                  Password
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-gray-700 text-white"
                  required
                />
              </div>

              {authError && (
                <div className="bg-red-900/50 border border-red-600 rounded-md p-3 text-sm text-red-300">
                  {authError}
                </div>
              )}

              <button
                type="submit"
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors font-medium"
              >
                {authMode === 'login' ? translate('authLogin', locale) : translate('authRegister', locale)}
              </button>
            </div>
          </form>
        </div>
      </div>
    )
  }

  // Main chat interface - check if we should show centered welcome screen
  const showCenteredWelcome = !currentSessionId;
  console.log('Render check:', { currentSessionId, showCenteredWelcome });

  // If no session, show centered welcome screen
  if (showCenteredWelcome) {
    return (
      <div className="relative h-screen min-h-0 bg-gray-900">
        {/* Expandable Left Sidebar - Always Visible */}
        <div className={`fixed left-0 top-0 h-full z-50 transition-all duration-300 ${isLeftSidebarExpanded ? 'w-80' : 'w-12'}`}>
          <div className="h-full bg-gray-800/90 backdrop-blur-sm border-r border-gray-700 flex flex-col">
            {!isLeftSidebarExpanded ? (
              /* Collapsed minimal sidebar */
              <div 
                className="h-full flex flex-col cursor-pointer"
                onClick={() => setIsLeftSidebarExpanded(true)}
              >
                {/* Top Section */}
                <div className="flex-1 p-2">
                  <div className="flex flex-col gap-2">
                    {/* Menu Button */}
                    <div className="flex items-center justify-center p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-white">
                      <Menu className="w-5 h-5" />
                    </div>
                    
                    {/* New Chat Button */}
                    <div 
                      className="flex items-center justify-center p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
                      onClick={(e) => {
                        e.stopPropagation()
                        createNewSession()
                      }}
                    >
                      <Plus className="w-5 h-5" />
                    </div>
                    
                    {/* Search Button */}
                    <div 
                      className="flex items-center justify-center p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
                      onClick={(e) => {
                        e.preventDefault()
                        e.stopPropagation()
                        console.log('Welcome screen search button clicked')
                        setShowSearchModal(true)
                      }}
                      title="Search chats (⌘K)"
                    >
                      <SearchIcon className="w-5 h-5" />
                    </div>
                  </div>
                </div>
                
                {/* Bottom Section - User Profile */}
                <div className="p-2 border-t border-gray-700">
                  <div className="relative" data-user-menu>
                    <div 
                      className="flex items-center justify-center p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
                      onClick={(e) => {
                        e.stopPropagation()
                        setShowUserMenu(!showUserMenu)
                      }}
                    >
                      <UserIcon className="w-5 h-5" />
                    </div>
                    
                    {/* User Menu Dropdown */}
                    {showUserMenu && (
                      <div className="absolute bottom-full left-12 mb-2 w-48 bg-gray-800 border border-gray-600 rounded-lg shadow-lg">
                        <div className="py-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              setShowSettings(true)
                              setShowUserMenu(false)
                            }}
                            className="w-full flex items-center px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                          >
                            <Settings className="w-4 h-4 mr-3" />
                            Settings
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              setShowLogoutConfirm(true)
                              setShowUserMenu(false)
                            }}
                            className="w-full flex items-center px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                          >
                            <LogOut className="w-4 h-4 mr-3" />
                            Log out
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              /* Expanded sidebar - ChatGPT style with labeled icons */
              <>
                {/* Top section with expanded icons */}
                <div className="p-2 space-y-1">
                  {/* Close/Menu Button */}
                  <div 
                    className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-300 hover:text-white cursor-pointer w-full"
                    onClick={() => setIsLeftSidebarExpanded(false)}
                  >
                    <Menu className="w-5 h-5 flex-shrink-0" />
                    <span className="text-sm font-medium truncate">Close sidebar</span>
                  </div>
                  
                  {/* New Chat Button */}
                  <div 
                    className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-300 hover:text-white cursor-pointer w-full"
                    onClick={createNewSession}
                  >
                    <Plus className="w-5 h-5 flex-shrink-0" />
                    <span className="text-sm font-medium truncate">New chat</span>
                  </div>
                  
                  {/* Search Button */}
                  <div 
                    className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-300 hover:text-white cursor-pointer w-full"
                    onClick={() => setShowSearchModal(true)}
                    title="Search chats (⌘K)"
                  >
                    <SearchIcon className="w-5 h-5 flex-shrink-0" />
                    <span className="text-sm font-medium truncate">Search chats</span>
                  </div>
                </div>

                {/* Chat Sessions List */}
                <div className="flex-1 overflow-y-auto">
                  <div className="px-2 pb-2">
                    <div className="text-xs font-medium text-gray-500 mb-2 px-2">Recent</div>
                    <div className="space-y-1">
                      {chatSessions.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-8 text-center">
                          <MessageCircle className="w-12 h-12 text-gray-600 mb-3" />
                          <p className="text-gray-400 text-sm">{translate('noChatsYet', locale)}</p>
                          <p className="text-gray-500 text-xs mt-1">
                            {translate('clickNewChat', locale)}
                          </p>
                        </div>
                      ) : (
                        chatSessions.map((session) => (
                          <SessionItem
                            key={session.id}
                            session={session}
                            isActive={currentSessionId === session.id}
                            onClick={() => switchToSession(session.id)}
                            onDelete={(e) => {
                              e.stopPropagation()
                              deleteSession(session.id)
                            }}
                          />
                        ))
                      )}
                    </div>
                  </div>
                </div>

                {/* Bottom section - User profile */}
                <div className="p-2 border-t border-gray-700">
                  <div className="relative" data-user-menu>
                    {/* User Profile Button */}
                    <div 
                      className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-300 hover:text-white cursor-pointer w-full"
                      onClick={(e) => {
                        e.stopPropagation()
                        setShowUserMenu(!showUserMenu)
                      }}
                    >
                      <UserIcon className="w-5 h-5 flex-shrink-0" />
                      <span className="text-sm font-medium truncate">{user.username}</span>
                    </div>
                    
                    {/* User Menu Dropdown */}
                    {showUserMenu && (
                      <div className="absolute bottom-full left-0 mb-2 w-full bg-gray-800 border border-gray-600 rounded-lg shadow-lg">
                        <div className="py-1">
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              setShowSettings(true)
                              setShowUserMenu(false)
                            }}
                            className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                          >
                            <Settings className="w-4 h-4 flex-shrink-0" />
                            <span>Settings</span>
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              setShowLogoutConfirm(true)
                              setShowUserMenu(false)
                            }}
                            className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                          >
                            <LogOut className="w-4 h-4 flex-shrink-0" />
                            <span>Log out</span>
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Centered Welcome Content */}
        <div className="absolute inset-0 flex flex-col bg-gray-900">

          {/* Centered Welcome Screen */}
          <div className="flex-1 flex flex-col items-center justify-center min-h-0 p-4 md:p-6 bg-gray-900">
            <div className="w-full max-w-2xl flex flex-col items-center justify-center text-center welcome-entrance">
              {/* App Icon and Title */}
              <div className="mb-6 md:mb-8">
                <div className="welcome-icon">
                  <TrendingUp className="w-8 h-8 text-white" />
                </div>
                <h1 className="text-2xl md:text-3xl font-bold text-white mb-2">
                  {translate('appTitle', locale)}
                </h1>
                <p className="text-sm md:text-base text-gray-400 mb-8 max-w-xl mx-auto leading-relaxed">
                  {translate('appSubtitle', locale)}
                </p>
              </div>

              {/* Main Input - Centered - Reusing Chat Interface Input */}
              <div className="w-full max-w-2xl mb-6">
                <form onSubmit={onSubmitChat} className="relative">
                  <div className="chat-input-container relative flex items-end gap-2 p-3 bg-gray-800/95 backdrop-blur-sm border border-gray-600 rounded-2xl shadow-lg hover:shadow-xl focus-within:shadow-xl focus-within:border-gray-400 focus-within:bg-gray-800 transition-all duration-200">
                    {/* Model Selector */}
                    {models && (
                      <div className="relative flex-shrink-0" data-model-dropdown>
                        <button
                          type="button"
                          onClick={() => setShowModelDropdown(!showModelDropdown)}
                          className="flex items-center justify-center w-8 h-8 hover:bg-gray-700 transition-colors rounded-lg"
                          title={`Current model: ${deployment ? (models.available.find(m => m.id === deployment)?.name || 'GPT-4') : 'GPT-4'}`}
                        >
                          <Cpu className="w-4 h-4 text-gray-400" />
                        </button>
                        
                        {/* Dropdown Menu */}
                        {showModelDropdown && (
                          <div className="absolute bottom-full left-0 mb-3 w-72 bg-gray-800 border border-gray-600 rounded-xl shadow-2xl backdrop-blur-xl z-50 max-h-80 overflow-y-auto">
                            <div className="p-3">
                              <div className="text-xs font-medium text-gray-400 mb-2 px-2">Select Model</div>
                              {models.available.map((model) => (
                                <button
                                  key={model.id}
                                  type="button"
                                  onClick={() => {
                                    setDeployment(model.id)
                                    setShowModelDropdown(false)
                                  }}
                                  disabled={!model.available}
                                  className={`w-full flex items-center justify-between px-3 py-2.5 text-left rounded-lg hover:bg-gray-700 transition-all duration-150 ${
                                    deployment === model.id ? 'bg-gray-700 ring-1 ring-blue-500/30' : ''
                                  } ${!model.available ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                                >
                                  <div className="flex items-center space-x-3">
                                    <div className={`w-2 h-2 rounded-full ${model.available ? 'bg-green-400' : 'bg-gray-500'}`}></div>
                                    <div>
                                      <div className="text-white text-sm font-medium">{model.name}</div>
                                      {model.default && (
                                        <div className="text-xs text-gray-400">Default</div>
                                      )}
                                    </div>
                                  </div>
                                  {deployment === model.id && (
                                    <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                                  )}
                                  {!model.available && (
                                    <span className="text-xs text-orange-400 bg-orange-400/20 px-2 py-1 rounded-md">max</span>
                                  )}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {/* Text Input */}
                    <div className="flex-1 min-w-0">
                      <textarea
                        id="chat-input-welcome"
                        name="chat-input-welcome"
                        ref={textareaRef}
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        onKeyDown={onComposerKeyDown}
                        placeholder={translate('placeholder', locale)}
                        className="w-full resize-none bg-transparent text-white placeholder-gray-500 focus:placeholder-gray-400 focus:outline-none leading-relaxed transition-colors duration-200"
                        rows={1}
                        style={{ 
                          minHeight: '20px', 
                          maxHeight: '160px',
                          paddingTop: '2px',
                          paddingBottom: '2px'
                        }}
                      />
                    </div>

                    {/* Action Buttons */}
                    <div className="flex items-end gap-1 flex-shrink-0">
                      {/* Send/Stop Button */}
                      {isStreaming ? (
                        <button
                          type="button"
                          onClick={onStopGenerating}
                          className="p-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-xl transition-all duration-200 hover:scale-105"
                          title="Stop generating"
                        >
                          <Square className="w-4 h-4" />
                        </button>
                      ) : (
                        <button
                          type="submit"
                          disabled={!prompt.trim() || chatLoading}
                          className="p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:hover:bg-gray-600 text-white rounded-xl transition-all duration-200 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                          title={translate('send', locale)}
                        >
                          {chatLoading ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Send className="w-4 h-4" />
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                </form>
              </div>



              {/* Network Status Indicator */}
              <div className="mb-4 w-full flex justify-center">
                <NetworkStatusIndicator 
                  isOnline={isOnline} 
                  connectionQuality={connectionQuality}
                />
              </div>

              {/* Error Display */}
              {chatError && (
                <div className="mb-4 w-full max-w-2xl">
                  <ErrorDisplay 
                    error={chatError} 
                    onRetry={onSubmitChat}
                    onDismiss={() => setChatError(null)}
                    canRetry={!!prompt.trim() && !chatLoading && !isStreaming}
                  />
                </div>
              )}

              {/* Quick actions or suggestions */}
              <div className="text-center quick-actions-fade">
                <p className="text-xs text-gray-400 mb-3">
                  Ask about stocks, market analysis, or financial insights
                </p>
                <div className="flex flex-wrap justify-center gap-1.5 text-xs">
                  <span className="px-2.5 py-1 bg-blue-800/50 text-blue-300 rounded-full hover:bg-blue-700/50 transition-colors cursor-default">Stock Analysis</span>
                  <span className="px-2.5 py-1 bg-green-800/50 text-green-300 rounded-full hover:bg-green-700/50 transition-colors cursor-default">Market Data</span>
                  <span className="px-2.5 py-1 bg-purple-800/50 text-purple-300 rounded-full hover:bg-purple-700/50 transition-colors cursor-default">Financial News</span>
                  <span className="px-2.5 py-1 bg-amber-800/50 text-amber-300 rounded-full hover:bg-amber-700/50 transition-colors cursor-default">Risk Assessment</span>
                </div>
              </div>
            </div>
          </div>


        </div>

        {/* Settings and Logout modals remain the same */}
        {/* Enhanced Settings Modal */}
        <Transition
          show={showSettings}
          enter="transition-opacity duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="transition-opacity duration-300"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        >
          <div className="bg-white rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">{translate('settings', locale)}</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="p-1 rounded-md hover:bg-gray-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-6">
              <div>
                <label htmlFor="system-prompt" className="block text-sm font-medium text-gray-700 mb-2">
                  {translate('systemPrompt', locale)}
                </label>
                <textarea
                  id="system-prompt"
                  name="system-prompt"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="w-full h-32 border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter system prompt..."
                />
              </div>

              {/* Language selector */}
              <div>
                <label htmlFor="language-select" className="block text-sm font-medium text-gray-700 mb-2">
                  {translate('language', locale)}
                </label>
                <select
                  id="language-select"
                  name="language-select"
                  value={locale}
                  onChange={(e) => {
                    const l = (e.target.value === 'ja' ? 'ja' : 'en') as Locale
                    setLocale(l)
                    setStoredLocale(l)
                  }}
                  className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="en">{translate('english', locale)}</option>
                  <option value="ja">{translate('japanese', locale)}</option>
                </select>
              </div>

              {models && (
                <div>
                  <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-2">
                    {translate('modelSelection', locale)}
                  </label>
                  <select
                    id="model-select"
                    name="model-select"
                    value={deployment}
                    onChange={(e) => setDeployment(e.target.value)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">{translate('selectModelPlaceholder', locale)}</option>
                    {models.available.map((model) => (
                      <option key={model.id} value={model.id} disabled={!model.available}>
                        {model.name} {model.default ? '(Default)' : ''} - {model.provider.toUpperCase()}
                      </option>
                    ))}
                  </select>

                  {deployment && (
                    <div className="mt-3 p-3 bg-gray-50 rounded-lg text-sm">
                      {(() => {
                        const selectedModel = models.available.find(m => m.id === deployment)
                        return selectedModel ? (
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="font-medium">{translate('selectedModel', locale)}</span>
                              <span>{selectedModel.name}</span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="font-medium">{translate('provider', locale)}</span>
                              <span className="uppercase">{selectedModel.provider}</span>
                            </div>
                            <div className="flex items-center justify-between">
                              <span className="font-medium">{translate('status', locale)}</span>
                              <span className={clsx(
                                "inline-flex items-center space-x-1",
                                selectedModel.available ? 'text-green-600' : 'text-red-600'
                              )}>
                                <span className={clsx(
                                  "inline-block w-2 h-2 rounded-full",
                                  selectedModel.available ? "bg-green-500" : "bg-red-500"
                                )} />
                                <span>{selectedModel.available ? translate('available', locale) : translate('unavailable', locale)}</span>
                              </span>
                            </div>
                            <div className="pt-2 border-t border-gray-200">
                              <span className="font-medium">{translate('description', locale)}</span>
                              <p className="text-gray-600 mt-1">{selectedModel.description}</p>
                            </div>
                          </div>
                        ) : <p className="text-red-600">{translate('unknownModel', locale)}</p>
                      })()}
                    </div>
                  )}

                  <div className="mt-4 p-3 bg-blue-50 rounded-lg text-sm">
                    <h4 className="font-medium text-blue-900 mb-2">{translate('availableModels', locale)}</h4>
                    <div className="space-y-1">
                      {models.available.filter(m => m.available).map((model) => (
                        <div key={model.id} className="flex items-center justify-between">
                          <span className="text-blue-800">{model.name}</span>
                          <span className="text-xs text-blue-600 uppercase">{model.provider}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </Transition>

        <Transition
          show={showLogoutConfirm}
          enter="transition-opacity duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="transition-opacity duration-300"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        >
          <div className="bg-white rounded-xl p-6 max-w-sm w-full mx-4">
            <h3 className="text-lg font-semibold mb-2">{translate('signOut', locale)}</h3>
            <p className="text-gray-600 mb-4">{translate('signOutConfirm', locale)}</p>
            <div className="flex space-x-3">
              <button
                onClick={() => setShowLogoutConfirm(false)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
              >
                {translate('cancel', locale)}
              </button>
              <button
                onClick={() => {
                  setShowLogoutConfirm(false)
                  onLogout()
                }}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
              >
                {translate('signOut', locale)}
              </button>
            </div>
          </div>
        </Transition>
      </div>
    )
  }

  // Regular chat interface with sidebar and messages
  return (
    <div className="relative h-screen bg-gray-900">
      {/* Expandable Left Sidebar - Always Visible */}
      <div className={`fixed left-0 top-0 h-full z-50 transition-all duration-300 ${isLeftSidebarExpanded ? 'w-80' : 'w-12'}`}>
        <div className="h-full bg-gray-800/90 backdrop-blur-sm border-r border-gray-700 flex flex-col">
          {!isLeftSidebarExpanded ? (
            /* Collapsed minimal sidebar */
            <div 
              className="h-full flex flex-col cursor-pointer"
              onClick={() => setIsLeftSidebarExpanded(true)}
            >
              {/* Top Section */}
              <div className="flex-1 p-2">
                <div className="flex flex-col gap-2">
                  {/* Menu Button */}
                  <div className="flex items-center justify-center p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-white">
                    <Menu className="w-5 h-5" />
                  </div>
                  
                  {/* New Chat Button */}
                  <div 
                    className="flex items-center justify-center p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
                    onClick={(e) => {
                      e.stopPropagation()
                      createNewSession()
                    }}
                  >
                    <Plus className="w-5 h-5" />
                  </div>
                  
                  {/* Search Button */}
                  <div 
                    className="flex items-center justify-center p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
                    onClick={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      console.log('Regular chat search button clicked')
                      setShowSearchModal(true)
                    }}
                    title="Search chats (⌘K)"
                  >
                    <SearchIcon className="w-5 h-5" />
                  </div>
                </div>
              </div>
              
              {/* Bottom Section - User Profile */}
              <div className="p-2 border-t border-gray-700">
                <div className="relative" data-user-menu>
                  <div 
                    className="flex items-center justify-center p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-white"
                    onClick={(e) => {
                      e.stopPropagation()
                      setShowUserMenu(!showUserMenu)
                    }}
                  >
                    <UserIcon className="w-5 h-5" />
                  </div>
                  
                  {/* User Menu Dropdown */}
                  {showUserMenu && (
                    <div className="absolute bottom-full left-12 mb-2 w-48 bg-gray-800 border border-gray-600 rounded-lg shadow-lg">
                      <div className="py-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setShowSettings(true)
                            setShowUserMenu(false)
                          }}
                          className="w-full flex items-center px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                        >
                          <Settings className="w-4 h-4 mr-3" />
                          Settings
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setShowLogoutConfirm(true)
                            setShowUserMenu(false)
                          }}
                          className="w-full flex items-center px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                        >
                          <LogOut className="w-4 h-4 mr-3" />
                          Log out
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            /* Expanded sidebar - ChatGPT style with labeled icons */
            <>
              {/* Top section with expanded icons */}
              <div className="p-2 space-y-1">
                {/* Close/Menu Button */}
                <div 
                  className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-300 hover:text-white cursor-pointer w-full"
                  onClick={() => setIsLeftSidebarExpanded(false)}
                >
                  <Menu className="w-5 h-5 flex-shrink-0" />
                  <span className="text-sm font-medium truncate">Close sidebar</span>
                </div>
                
                {/* New Chat Button */}
                <div 
                  className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-300 hover:text-white cursor-pointer w-full"
                  onClick={createNewSession}
                >
                  <Plus className="w-5 h-5 flex-shrink-0" />
                  <span className="text-sm font-medium truncate">New chat</span>
                </div>
                
                {/* Search Button */}
                <div 
                  className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-300 hover:text-white cursor-pointer w-full"
                  onClick={() => setShowSearchModal(true)}
                  title="Search chats (⌘K)"
                >
                  <SearchIcon className="w-5 h-5 flex-shrink-0" />
                  <span className="text-sm font-medium truncate">Search chats</span>
                </div>
              </div>

              {/* Chat Sessions List */}
              <ErrorBoundary>
                <div className="flex-1 overflow-y-auto">
                  <div className="px-2 pb-2">
                    <div className="text-xs font-medium text-gray-500 mb-2 px-2">Recent</div>
                    <div className="space-y-1">
                      {chatSessions.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-8 text-center">
                          <MessageCircle className="w-12 h-12 text-gray-600 mb-3" />
                          <p className="text-gray-400 text-sm">{translate('noChatsYet', locale)}</p>
                          <p className="text-gray-500 text-xs mt-1">
                            {translate('clickNewChat', locale)}
                          </p>
                        </div>
                      ) : (
                        chatSessions.map((session) => (
                          <ErrorBoundary key={session.id}>
                            <SessionItem
                              session={session}
                              isActive={currentSessionId === session.id}
                              onClick={() => switchToSession(session.id)}
                              onDelete={(e) => {
                                e.stopPropagation()
                                deleteSession(session.id)
                              }}
                            />
                          </ErrorBoundary>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              </ErrorBoundary>

              {/* Bottom section - User profile */}
              <div className="p-2 border-t border-gray-700">
                <div className="relative" data-user-menu>
                  {/* User Profile Button */}
                  <div 
                    className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-700 transition-colors text-gray-300 hover:text-white cursor-pointer w-full"
                    onClick={(e) => {
                      e.stopPropagation()
                      setShowUserMenu(!showUserMenu)
                    }}
                  >
                    <UserIcon className="w-5 h-5 flex-shrink-0" />
                    <span className="text-sm font-medium truncate">{user.username}</span>
                  </div>
                  
                  {/* User Menu Dropdown */}
                  {showUserMenu && (
                    <div className="absolute bottom-full left-0 mb-2 w-full bg-gray-800 border border-gray-600 rounded-lg shadow-lg">
                      <div className="py-1">
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setShowSettings(true)
                            setShowUserMenu(false)
                          }}
                          className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                        >
                          <Settings className="w-4 h-4 flex-shrink-0" />
                          <span>Settings</span>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setShowLogoutConfirm(true)
                            setShowUserMenu(false)
                          }}
                          className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                        >
                          <LogOut className="w-4 h-4 flex-shrink-0" />
                          <span>Log out</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="absolute inset-0 flex flex-col bg-gray-900">
        {/* Messages Container with Sticky Header */}
        <ErrorBoundary>
          <div ref={messagesContainerRef} className="flex-1 overflow-y-auto messages-scroll-container">

            
            {/* Messages Content */}
            <div className="p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center max-w-md">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Bot className="w-8 h-8 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{translate('onboardingTitle', locale)}</h3>
                <p className="text-gray-400 mb-4">{translate('onboardingBody', locale)}</p>
                {models && deployment && (
                  <div className="inline-flex items-center space-x-2 px-3 py-1 bg-blue-800/50 rounded-full text-sm text-blue-300">
                    <Cpu className="w-4 h-4" />
                    <span>{translate('usingModel', locale)} {models.available.find(m => m.id === deployment)?.name || 'AI Model'}</span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <>
              {messages.reduce((acc: React.ReactNode[], message, index) => {
                if (message.role === 'user') {
                  // Find the corresponding assistant message
                  const assistantMessage = messages[index + 1]?.role === 'assistant' ? messages[index + 1] : undefined
                  
                  // Determine if this is currently streaming (only for the last assistant message)
                  const isCurrentlyStreaming = isStreaming && assistantMessage && index === messages.length - 2
                  
                  acc.push(
                    <QASection
                      key={message.id}
                      userMessage={message}
                      assistantMessage={assistantMessage}
                      isActive={false}
                      activeTools={isCurrentlyStreaming ? activeTools : []}
                      isStreaming={isCurrentlyStreaming}
                      currentTab={assistantMessage ? getCurrentTab(assistantMessage.id) : 'answer'}
                      onTabChange={(tab) => {
                        if (assistantMessage) {
                          setActiveTab(prev => ({ ...prev, [assistantMessage.id]: tab }))
                        }
                      }}
                      messagesContainerRef={messagesContainerRef}
                    />
                  )
                }
                return acc
              }, [])}

              {isTyping && !(messages.length > 0 && messages[messages.length - 1]?.role === 'assistant' && (!messages[messages.length - 1]?.content || messages[messages.length - 1]?.content.trim() === '')) && (
                <div className="max-w-2xl mx-auto px-8 py-4">
                  <div className="flex items-center gap-2 text-gray-400">
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" />
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              )}

              {/* Live Tool Activity Panel */}
              {activeTools.length > 0 && (
                <div className="max-w-2xl mx-auto px-8 py-2">
                  <div className="bg-gray-800 border border-blue-600 rounded-lg px-4 py-3 shadow-sm">
                    <p className="text-xs font-medium text-blue-300 mb-2">{translate('toolActivity', locale)}</p>
                    <div className="space-y-1">
                      {activeTools.map((t) => {
                        // Get tool icon and description based on tool name
                        const getToolInfo = (toolName: string) => {
                          const name = toolName.toLowerCase();
                          if (name.includes('web_search') || name === 'web_search') {
                            return { 
                              icon: <Globe className="w-3 h-3" />, 
                              label: 'Web Search',
                              color: 'bg-green-50 text-green-700 border-green-100'
                            };
                          }
                          if (name.includes('web_search_news')) {
                            return { 
                              icon: <Newspaper className="w-3 h-3" />, 
                              label: 'News Search',
                              color: 'bg-purple-50 text-purple-700 border-purple-100'
                            };
                          }
                          if (name.includes('augmented_rag') || name.includes('financial_context')) {
                            return { 
                              icon: <Database className="w-3 h-3" />, 
                              label: 'Enhanced Search',
                              color: 'bg-indigo-50 text-indigo-700 border-indigo-100'
                            };
                          }
                          if (name.includes('rag_search')) {
                            return { 
                              icon: <BookOpen className="w-3 h-3" />, 
                              label: 'Knowledge Base',
                              color: 'bg-blue-50 text-blue-700 border-blue-100'
                            };
                          }
                          if (name.includes('stock') || name.includes('market') || name.includes('financial')) {
                            return { 
                              icon: <LineChart className="w-3 h-3" />, 
                              label: 'Market Data',
                              color: 'bg-orange-50 text-orange-700 border-orange-100'
                            };
                          }
                          // Default
                          return { 
                            icon: <Wrench className="w-3 h-3" />, 
                            label: toolName,
                            color: 'bg-gray-50 text-gray-700 border-gray-100'
                          };
                        };

                        const toolInfo = getToolInfo(t.name);

                        return (
                          <div key={t.name} className="flex items-center gap-2 text-xs">
                            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border ${toolInfo.color}`}>
                              {toolInfo.icon}
                              {toolInfo.label}
                            </span>
                            {t.status === 'running' && (
                              <span className="inline-flex items-center text-gray-500">
                                <Loader2 className="w-3 h-3 mr-1 animate-spin" /> 
                                {translate('statusRunning', locale)}
                              </span>
                            )}
                            {t.status === 'completed' && (
                              <span className="inline-flex items-center text-green-600">
                                <Check className="w-3 h-3 mr-1" /> 
                                {translate('statusCompleted', locale)}
                              </span>
                            )}
                            {t.status === 'error' && (
                              <span className="inline-flex items-center text-red-600">
                                <XCircle className="w-3 h-3 mr-1" /> 
                                {translate('statusError', locale)}
                              </span>
                            )}
                            {t.error && (
                              <span className="text-red-500">{String(t.error).slice(0, 120)}</span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* Cached response indicator */}
              {isCachedStream && (
                <div className="flex space-x-3 justify-start">
                  <div className="w-8 h-8 bg-gray-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 3v18h18"/></svg>
                  </div>
                  <div className="bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-xs text-gray-300">{translate('cachedNotice', locale)}</div>
                </div>
              )}
            </>
          )}

              <div ref={endRef} />
            </div>
          </div>
        </ErrorBoundary>

        {/* Network status indicator */}
        <div className="mx-4 mb-2">
          <NetworkStatusIndicator 
            isOnline={isOnline} 
            connectionQuality={connectionQuality}
          />
        </div>

        {chatError && (
          <ErrorDisplay 
            error={chatError} 
            onRetry={onSubmitChat}
            onDismiss={() => setChatError(null)}
            canRetry={!!prompt.trim() && !chatLoading && !isStreaming}
          />
        )}

        {/* Chat Input */}
        <div className="bg-gray-900 p-6 border-t border-gray-800">
          <div className="max-w-2xl mx-auto">
            <form onSubmit={onSubmitChat} className="relative">
              <div className="chat-input-container relative flex items-end gap-2 p-3 bg-gray-800/95 backdrop-blur-sm border border-gray-600 rounded-2xl shadow-lg hover:shadow-xl focus-within:shadow-xl focus-within:border-gray-400 focus-within:bg-gray-800 transition-all duration-200">
                {/* Model Selector */}
                {models && (
                  <div className="relative flex-shrink-0" data-model-dropdown>
                    <button
                      type="button"
                      onClick={() => setShowModelDropdown(!showModelDropdown)}
                      className="flex items-center justify-center w-8 h-8 hover:bg-gray-700 transition-colors rounded-lg"
                      title={`Current model: ${deployment ? (models.available.find(m => m.id === deployment)?.name || 'GPT-4') : 'GPT-4'}`}
                    >
                      <Cpu className="w-4 h-4 text-gray-400" />
                    </button>
                    
                    {/* Dropdown Menu */}
                    {showModelDropdown && (
                      <div className="absolute bottom-full left-0 mb-3 w-72 bg-gray-800 border border-gray-600 rounded-xl shadow-2xl backdrop-blur-xl z-50 max-h-80 overflow-y-auto">
                        <div className="p-3">
                          <div className="text-xs font-medium text-gray-400 mb-2 px-2">Select Model</div>
                          {models.available.map((model) => (
                            <button
                              key={model.id}
                              type="button"
                              onClick={() => {
                                setDeployment(model.id)
                                setShowModelDropdown(false)
                              }}
                              disabled={!model.available}
                              className={`w-full flex items-center justify-between px-3 py-2.5 text-left rounded-lg hover:bg-gray-700 transition-all duration-150 ${
                                deployment === model.id ? 'bg-gray-700 ring-1 ring-blue-500/30' : ''
                              } ${!model.available ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                            >
                              <div className="flex items-center space-x-3">
                                <div className={`w-2 h-2 rounded-full ${model.available ? 'bg-green-400' : 'bg-gray-500'}`}></div>
                                <div>
                                  <div className="text-white text-sm font-medium">{model.name}</div>
                                  {model.default && (
                                    <div className="text-xs text-gray-400">Default</div>
                                  )}
                                </div>
                              </div>
                              {deployment === model.id && (
                                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                              )}
                              {!model.available && (
                                <span className="text-xs text-orange-400 bg-orange-400/20 px-2 py-1 rounded-md">max</span>
                              )}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                
                {/* Text Input */}
                <div className="flex-1 min-w-0">
                  <textarea
                    id="chat-input-main"
                    name="chat-input-main"
                    ref={textareaRef}
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={onComposerKeyDown}
                    placeholder={currentSessionId ? translate('placeholder', locale) : 'Create a new chat to start messaging...'}
                    disabled={!currentSessionId}
                    className="w-full resize-none bg-transparent text-white placeholder-gray-500 focus:placeholder-gray-400 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50 leading-relaxed transition-colors duration-200"
                    rows={1}
                    style={{ 
                      minHeight: '20px', 
                      maxHeight: '160px',
                      paddingTop: '2px',
                      paddingBottom: '2px'
                    }}
                  />
                </div>

                {/* Action Buttons */}
                <div className="flex items-end gap-1 flex-shrink-0">
                  {/* Additional action buttons could go here (like attachment, etc.) */}
                  
                  {/* Send/Stop Button */}
                  {isStreaming ? (
                    <button
                      type="button"
                      onClick={onStopGenerating}
                      className="p-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-xl transition-all duration-200 hover:scale-105"
                      title="Stop generating"
                    >
                      <Square className="w-4 h-4" />
                    </button>
                  ) : (
                    <button
                      type="submit"
                      disabled={!currentSessionId || !prompt.trim() || chatLoading}
                      className="p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:hover:bg-gray-600 text-white rounded-xl transition-all duration-200 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                      title={translate('send', locale)}
                    >
                      {chatLoading ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Send className="w-4 h-4" />
                      )}
                    </button>
                  )}
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>



      {/* Enhanced Settings Modal */}
      <Transition
        show={showSettings}
        enter="transition-opacity duration-300"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-300"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      >
        <div className="bg-gray-800 rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto border border-gray-600">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-white">{translate('settings', locale)}</h3>
            <button
              onClick={() => setShowSettings(false)}
              className="p-1 rounded-md hover:bg-gray-700 text-gray-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                {translate('systemPrompt', locale)}
              </label>
              <textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                className="w-full h-32 bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-sm text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter system prompt..."
              />
            </div>

            {/* Language selector */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                {translate('language', locale)}
              </label>
              <select
                value={locale}
                onChange={(e) => {
                  const l = (e.target.value === 'ja' ? 'ja' : 'en') as Locale
                  setLocale(l)
                  setStoredLocale(l)
                }}
                className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="en">{translate('english', locale)}</option>
                <option value="ja">{translate('japanese', locale)}</option>
              </select>
            </div>

            {models && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  {translate('modelSelection', locale)}
                </label>
                <select
                  value={deployment}
                  onChange={(e) => setDeployment(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="" className="bg-gray-700">{translate('selectModelPlaceholder', locale)}</option>
                  {models.available.map((model) => (
                    <option key={model.id} value={model.id} disabled={!model.available} className="bg-gray-700">
                      {model.name} {model.default ? '(Default)' : ''} - {model.provider.toUpperCase()}
                    </option>
                  ))}
                </select>

                {deployment && (
                  <div className="mt-3 p-3 bg-gray-700 border border-gray-600 rounded-lg text-sm text-white">
                    {(() => {
                      const selectedModel = models.available.find(m => m.id === deployment)
                      return selectedModel ? (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{translate('selectedModel', locale)}</span>
                            <span>{selectedModel.name}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{translate('provider', locale)}</span>
                            <span className="uppercase">{selectedModel.provider}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="font-medium">{translate('status', locale)}</span>
                            <span className={clsx(
                              "inline-flex items-center space-x-1",
                              selectedModel.available ? 'text-green-600' : 'text-red-600'
                            )}>
                              <span className={clsx(
                                "inline-block w-2 h-2 rounded-full",
                                selectedModel.available ? "bg-green-500" : "bg-red-500"
                              )} />
                              <span>{selectedModel.available ? translate('available', locale) : translate('unavailable', locale)}</span>
                            </span>
                          </div>
                          <div className="pt-2 border-t border-gray-600">
                            <span className="font-medium text-gray-300">{translate('description', locale)}</span>
                            <p className="text-gray-400 mt-1">{selectedModel.description}</p>
                          </div>
                        </div>
                      ) : <p className="text-red-400">{translate('unknownModel', locale)}</p>
                    })()}
                  </div>
                )}

                <div className="mt-4 p-3 bg-gray-700 border border-gray-600 rounded-lg text-sm">
                  <h4 className="font-medium text-blue-400 mb-2">{translate('availableModels', locale)}</h4>
                  <div className="space-y-1">
                    {models.available.filter(m => m.available).map((model) => (
                      <div key={model.id} className="flex items-center justify-between">
                        <span className="text-gray-200">{model.name}</span>
                        <span className="text-xs text-blue-400 uppercase">{model.provider}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </Transition>

      <Transition
        show={showLogoutConfirm}
        enter="transition-opacity duration-300"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-300"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      >
        <div className="bg-gray-800 border border-gray-600 rounded-xl p-6 max-w-sm w-full mx-4">
          <h3 className="text-lg font-semibold text-white mb-2">{translate('signOut', locale)}</h3>
          <p className="text-gray-300 mb-4">{translate('signOutConfirm', locale)}</p>
          <div className="flex space-x-3">
            <button
              onClick={() => setShowLogoutConfirm(false)}
              className="flex-1 px-4 py-2 border border-gray-600 text-gray-300 rounded-md hover:bg-gray-700 hover:text-white transition-colors"
            >
              {translate('cancel', locale)}
            </button>
            <button
              onClick={() => {
                setShowLogoutConfirm(false)
                onLogout()
              }}
              className="flex-1 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              {translate('signOut', locale)}
            </button>
          </div>
        </div>
      </Transition>

      {/* Search Modal - Available in both welcome screen and regular chat */}
      <Transition
        show={showSearchModal}
        enter="transition-opacity duration-300"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="transition-opacity duration-300"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      >
        <div className="bg-gray-800 rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto border border-gray-600">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h3 className="text-lg font-semibold text-white">Search Conversations</h3>
              <p className="text-xs text-gray-400 mt-1">Press <kbd className="px-1.5 py-0.5 bg-gray-700 rounded text-xs">⌘K</kbd> to search anytime</p>
            </div>
            <button
              onClick={() => {
                setShowSearchModal(false)
                setSearchModalQuery('')
              }}
              className="p-1 rounded-md hover:bg-gray-700 text-gray-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="mb-4">
            <input
              id="search-conversations"
              name="search-conversations"
              type="text"
              value={searchModalQuery}
              onChange={(e) => setSearchModalQuery(e.target.value)}
              placeholder="Search your conversations..."
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              autoFocus
            />
          </div>

          <div className="space-y-2 max-h-96 overflow-y-auto">
            {(() => {
              const query = searchModalQuery.toLowerCase().trim()
              const filteredSessions = query 
                ? chatSessions.filter(session => 
                    session.title.toLowerCase().includes(query) || 
                    session.lastMessage.toLowerCase().includes(query)
                  )
                : chatSessions.slice(0, 10) // Show recent 10 if no search

              if (filteredSessions.length === 0 && query) {
                return (
                  <div className="text-center py-8 text-gray-400">
                    <SearchIcon className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No conversations found for "{searchModalQuery}"</p>
                  </div>
                )
              }

              if (filteredSessions.length === 0 && !query) {
                return (
                  <div className="text-center py-8 text-gray-400">
                    <MessageCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No conversations yet</p>
                    <p className="text-xs mt-1">Start a new chat to begin</p>
                  </div>
                )
              }

              return filteredSessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => {
                    switchToSession(session.id)
                    setShowSearchModal(false)
                    setSearchModalQuery('')
                  }}
                  className="w-full text-left p-3 rounded-lg hover:bg-gray-700 transition-colors border border-transparent hover:border-gray-600"
                >
                  <div className="flex items-center justify-between">
                    <div className="min-w-0 flex-1">
                      <h4 className="text-sm font-medium text-white truncate">
                        {session.title}
                      </h4>
                      <p className="text-xs text-gray-400 truncate mt-1">
                        {session.lastMessage || 'No messages yet'}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {session.timestamp.toLocaleDateString()} • {session.messageCount} messages
                      </p>
                    </div>
                    <div className="ml-3 flex-shrink-0">
                      <MessageCircle className="w-4 h-4 text-gray-500" />
                    </div>
                  </div>
                </button>
              ))
            })()}
          </div>

          {!searchModalQuery && (
            <div className="mt-4 text-xs text-gray-500 text-center">
              Showing recent conversations. Start typing to search.
            </div>
          )}
        </div>
      </Transition>
    </div>
  )
}
