export const apiBase = (): string => {
  const fromEnv = (import.meta as any).env?.VITE_API_BASE as string | undefined
  if (import.meta.env.DEV) {
    const base = (fromEnv && fromEnv.trim()) || 'http://127.0.0.1:8000'
    console.log('[apiBase] DEV mode, returning:', base)
    return base
  }
  const base = (fromEnv && fromEnv.trim()) || window.location.origin
  console.log('[apiBase] PROD mode, returning:', base)
  return base
}

const apiKey = (import.meta as any).env?.VITE_APP_API_KEY as string | undefined

// Auth token management
const TOKEN_KEY = 'auth_token'
export type User = { id: number; username: string; email?: string | null; is_admin?: boolean }

export function getToken(): string | null {
  try { return localStorage.getItem(TOKEN_KEY) } catch { return null }
}
export function setToken(token: string | null) {
  try {
    if (!token) localStorage.removeItem(TOKEN_KEY)
    else localStorage.setItem(TOKEN_KEY, token)
  } catch {}
}

export function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = { ...(extra || {}) }
  const t = getToken()
  if (t) headers['Authorization'] = `Bearer ${t}`
  if (apiKey && apiKey.trim()) headers['X-API-Key'] = apiKey.trim()
  return headers
}

async function tryRefresh(): Promise<User | null> {
  // Uses HttpOnly cookie; must include credentials
  const res = await fetch(`${apiBase()}/auth/refresh`, { method: 'POST', credentials: 'include' })
  if (!res.ok) return null
  const data: LoginResponse = await res.json()
  setToken(data.access_token)
  return data.user
}

// Helper to create user-friendly error messages from network errors
function createUserFriendlyError(error: any, url: string): Error {
  // Handle network/fetch errors
  if (error instanceof TypeError && error.message.includes('fetch')) {
    return new Error('Connection failed')
  }
  if (error instanceof TypeError && (error.message.includes('NetworkError') || error.message.includes('Failed to fetch'))) {
    return new Error('Unable to reach server')
  }
  if (error.name === 'AbortError') {
    return new Error('Request cancelled')
  }
  if (error.message && error.message.toLowerCase().includes('timeout')) {
    return new Error('Request timed out')
  }
  
  // Return original error if it's already user-friendly
  return error instanceof Error ? error : new Error(String(error))
}

// Helper to determine if an error is retryable
function isRetryableError(error: any): boolean {
  if (error instanceof TypeError && (
    error.message.includes('fetch') || 
    error.message.includes('NetworkError') || 
    error.message.includes('Failed to fetch')
  )) {
    return true
  }
  if (error.message && error.message.toLowerCase().includes('timeout')) {
    return true
  }
  return false
}

// Helper for exponential backoff retry logic
async function withRetry<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 1000
): Promise<T> {
  let lastError: any
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await operation()
    } catch (error) {
      lastError = error
      
      // Don't retry if it's the last attempt or if the error is not retryable
      if (attempt === maxRetries || !isRetryableError(error)) {
        break
      }
      
      // Calculate delay with exponential backoff and jitter
      const delay = baseDelayMs * Math.pow(2, attempt) + Math.random() * 500
      console.log(`Request failed (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay.toFixed(0)}ms...`, error)
      
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }
  
  throw lastError
}

async function fetchJson<T>(url: string, init: RequestInit & { retryOn401?: boolean, maxRetries?: number } = {}): Promise<T> {
  const { retryOn401, maxRetries = 3, ...rest } = init
  
  return withRetry(async () => {
    try {
      const res = await fetch(url, rest)
      if (res.status === 401 && retryOn401) {
        const u = await tryRefresh()
        if (u) {
          const again = await fetch(url, { ...rest, headers: authHeaders((rest.headers as any) || {}) })
          if (!again.ok) {
            let detail = `${again.status} ${again.statusText}`
            try { const j = await again.json(); detail = (j as any).detail || JSON.stringify(j) } catch {}
            throw new Error(detail)
          }
          return again.json()
        }
      }
      if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`
        try { const j = await res.json(); detail = (j as any).detail || JSON.stringify(j) } catch {}
        throw new Error(detail)
      }
      return res.json()
    } catch (error) {
      throw createUserFriendlyError(error, url)
    }
  }, maxRetries)
}

export type ChatRequest = {
  prompt: string
  system_prompt?: string
  deployment?: string
  conversation_id?: string
  reset?: boolean
  stream?: boolean  // Add streaming support
  locale?: 'en' | 'ja'
  web_search_enabled?: boolean  // Whether to enable web search tools
}

export type ChatResponse = {
  content: string
  tool_calls?: Array<{ id: string; name: string; result: unknown }>
  conversation_id?: string
}

export async function chat(req: ChatRequest): Promise<ChatResponse> {
  return fetchJson(`${apiBase()}/chat`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(req),
    retryOn401: true,
  })
}

// Add streaming chat function
export async function chatStream(
  req: ChatRequest,
  onChunk: (chunk: { type: string; delta?: string; error?: string; [key: string]: any }) => void,
  onComplete: (data: { conversation_id?: string; tool_calls?: any[] }) => void,
  onError: (error: string) => void
): Promise<void> {
  // Module-level controller so callers can cancel the active stream
  // Only one stream is expected at a time in the UI.
  let abortController: AbortController | null = null
  ;(window as any).__chatStreamAbortController = abortController
  // Track completion in case the stream ends without an explicit "done" event
  let didComplete = false

  async function startStream(): Promise<Response> {
    abortController = new AbortController()
    ;(window as any).__chatStreamAbortController = abortController
    
    return withRetry(async () => {
      if (!abortController) {
        abortController = new AbortController()
        ;(window as any).__chatStreamAbortController = abortController
      }
      
      return fetch(`${apiBase()}/chat/stream`, {
        method: 'POST',
        headers: authHeaders({
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }),
        body: JSON.stringify(req),
        credentials: 'include',
        signal: abortController.signal,
      })
    }, 2) // Fewer retries for streaming to avoid long delays
  }

  try {
    // First attempt
    let response = await startStream()

    // Handle 401 by refreshing token once
    if (response.status === 401) {
      try {
        await refresh()
        response = await startStream()
      } catch {
        // Ignore refresh failure, will surface original 401
      }
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => '')
      onError(`HTTP ${response.status}: ${errorText || response.statusText}`)
      return
    }

    const contentType = response.headers.get('content-type') || ''
    if (!contentType.includes('text/event-stream')) {
      console.warn('Expected text/event-stream, got:', contentType)
    }

    const reader = response.body?.getReader()
    if (!reader) {
      onError('Failed to get response reader')
      return
    }

    const decoder = new TextDecoder()
    let buffer = ''
    let sawContentOrStart = false

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        // Normalize CRLF to LF
        buffer = buffer.replace(/\r\n/g, '\n')

        // Split on SSE message boundary
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          const lines = part.split('\n')
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6).trim()
              if (!jsonStr) continue
              try {
                const data = JSON.parse(jsonStr)
                if (data.type === 'error') {
                  onError(data.error)
                  return
                } else if (data.type === 'done') {
                  didComplete = true
                  onComplete(data)
                  return
                } else {
                  // Mark that we saw some valid stream content
                  sawContentOrStart = true
                  onChunk(data)
                }
              } catch (e) {
                console.warn('Failed to parse SSE data:', line, e)
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock()
    }

    // If the stream ended naturally without a 'done' event, finalize once
    if (!didComplete) {
      didComplete = true
      try {
        onComplete({})
      } catch (e) {
        // If consumer expects completion to always succeed, convert to error
        if (!sawContentOrStart) {
          onError('Stream ended unexpectedly')
        }
      }
    }
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      console.log('Stream was aborted')
      return
    }
    
    // Create user-friendly error message for streaming failures
    const friendlyError = createUserFriendlyError(error, `${apiBase()}/chat/stream`)
    onError(friendlyError.message)
  } finally {
    abortController = null
    ;(window as any).__chatStreamAbortController = null
  }
}

// Allow UI to cancel the current streaming request
export function cancelChatStream(): boolean {
  const ctrl = (window as any).__chatStreamAbortController as AbortController | null | undefined
  if (ctrl && typeof ctrl.abort === 'function') {
    try { ctrl.abort() } catch {}
    ;(window as any).__chatStreamAbortController = null
    return true
  }
  return false
}

export type ClearChatResponse = { conversation_id: string; cleared: boolean }

export async function clearChat(conversation_id: string): Promise<ClearChatResponse> {
  return fetchJson(`${apiBase()}/chat/clear`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ conversation_id }),
    retryOn401: true,
  })
}

export type ChatHistoryMessage = { role: string; content: string }
export type ChatHistoryResponse = { conversation_id: string; found: boolean; messages: ChatHistoryMessage[] }

export async function getChatHistory(
  conversation_id: string,
  opts?: { include_system?: boolean; max_messages?: number }
): Promise<ChatHistoryResponse> {
  const params = new URLSearchParams()
  if (opts?.include_system) params.set('include_system', 'true')
  if (typeof opts?.max_messages === 'number') params.set('max_messages', String(opts.max_messages))
  const qs = params.toString()
  const url = `${apiBase()}/chat/history/${encodeURIComponent(conversation_id)}${qs ? `?${qs}` : ''}`
  return fetchJson(url, { headers: authHeaders(), retryOn401: true })
}

// Models
export type ModelInfo = {
  id: string
  name: string
  description: string
  provider: string
  available: boolean
  default: boolean
}

export type ModelsResponse = {
  default: string
  available: ModelInfo[]
  total_count: number
  available_count: number
}

export async function listModels(): Promise<ModelsResponse> {
  return fetchJson(`${apiBase()}/chat/models`, { headers: authHeaders(), retryOn401: true })
}

// Auth endpoints
export type RegisterRequest = { username: string; password: string; email?: string }
export type RegisterResponse = { id: number; username: string; email?: string }
export async function register(req: RegisterRequest): Promise<RegisterResponse> {
  return fetchJson(`${apiBase()}/auth/register`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(req),
  })
}

export type LoginResponse = { access_token: string; token_type: string; user: User }
export async function login(username: string, password: string): Promise<LoginResponse> {
  const body = new URLSearchParams()
  body.set('username', username)
  body.set('password', password)
  
  try {
    const res = await fetch(`${apiBase()}/auth/token`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/x-www-form-urlencoded' }),
      body,
      credentials: 'include',
    })
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`
      try { const j = await res.json(); detail = (j as any).detail || JSON.stringify(j) } catch {}
      throw new Error(`Login failed: ${detail}`)
    }
    const data: LoginResponse = await res.json()
    setToken(data.access_token)
    return data
  } catch (error) {
    throw createUserFriendlyError(error, `${apiBase()}/auth/token`)
  }
}

export async function refresh(): Promise<LoginResponse> {
  const res = await fetch(`${apiBase()}/auth/refresh`, { method: 'POST', credentials: 'include' })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try { const j = await res.json(); detail = (j as any).detail || JSON.stringify(j) } catch {}
    throw new Error(`Refresh failed: ${detail}`)
  }
  const data: LoginResponse = await res.json()
  setToken(data.access_token)
  return data
}

export async function logout(): Promise<void> {
  await fetchJson(`${apiBase()}/auth/logout`, { method: 'POST', headers: authHeaders(), credentials: 'include' as any, retryOn401: true })
  setToken(null)
}

export async function me(): Promise<User> {
  return fetchJson(`${apiBase()}/auth/me`, { headers: authHeaders(), retryOn401: true })
}

// Password reset
export async function requestPasswordReset(username: string): Promise<{ ok: boolean; reset_token?: string; expires_at?: string }>{
  return fetchJson(`${apiBase()}/auth/request-password-reset`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ username }),
  })
}

export async function resetPassword(token: string, new_password: string): Promise<{ ok: boolean }>{
  return fetchJson(`${apiBase()}/auth/reset-password`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ token, new_password }),
  })
}

// Admin logs
export type AdminLogItem = {
  id: number
  user_id: number
  username?: string | null
  created_at?: string
  action: string
  conversation_id?: string | null
  prompt?: string | null
  response?: string | null
  tool_calls?: unknown
}
export type AdminLogsResponse = { total: number; items: AdminLogItem[] }

export async function getAdminLogs(params?: { limit?: number; offset?: number; user?: string; action?: string; conversation_id?: string }): Promise<AdminLogsResponse> {
  const q = new URLSearchParams()
  if (params?.limit) q.set('limit', String(params.limit))
  if (params?.offset) q.set('offset', String(params.offset))
  if (params?.user) q.set('user', params.user)
  if (params?.action) q.set('action', params.action)
  if (params?.conversation_id) q.set('conversation_id', params.conversation_id)
  const url = `${apiBase()}/admin/logs${q.toString() ? `?${q.toString()}` : ''}`
  return fetchJson(url, { headers: authHeaders(), retryOn401: true })
}
