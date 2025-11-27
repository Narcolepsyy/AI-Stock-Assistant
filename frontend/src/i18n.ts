// Simple i18n helper with English and Japanese, persisted in localStorage
import { useMemo, useCallback } from 'react'
export type Locale = 'en' | 'ja'

const STORAGE_KEY = 'app_locale'

// Cache for memoizing translations
const translationCache = new Map<string, string>()

const dict = {
  en: {
    appTitle: 'AI Stocks Assistant',
    appSubtitle: 'Your intelligent financial companion',
    authLogin: 'Sign In',
    authRegister: 'Create Account',
    settings: 'Settings',
    language: 'Language',
    selectLanguage: 'Select Language',
    english: 'English',
    japanese: '日本語',
    signOut: 'Sign Out',
    cancel: 'Cancel',
    closeSidebar: 'Close sidebar',
    openSidebar: 'Open sidebar',
    searchChats: 'Search chats...',
    noChatsFound: 'No chats found',
    tryDifferentSearch: 'Try a different search term',
    noChatsYet: 'No chats yet',
    clickNewChat: 'Click "New Chat" to start a conversation',
    modelSelection: 'AI Model Selection',
    systemPrompt: 'System Prompt',
    placeholder: 'Ask about stocks, market data, or financial analysis...',
    // Onboarding/empty state
    onboardingTitle: 'Welcome to AI Stocks Assistant',
    onboardingBody: 'Ask me anything about stocks, market data, financial analysis, or investment strategies. I can help you with real-time quotes, historical data, company profiles, news, and risk assessments.',
    usingModel: 'Using',
    // Tool activity panel
    toolActivity: 'Tool activity',
    statusRunning: 'running',
    statusCompleted: 'completed',
    statusError: 'error',
    // Cached indicator
    cachedNotice: 'Served from cache — tools may not run for cached responses.',
    // Errors
    mustSignIn: 'Please sign in to chat.',
    loginFailed: 'Login failed',
    registerFailed: 'Register failed',
    streamingFailed: 'Streaming chat failed',
    stop: 'Stop',
    send: 'Send',
    newChat: 'New Chat',
    messagesCountSuffix: 'messages',
    responding: 'Responding...',
    // Settings details
    selectAIModelPlaceholder: 'Select AI Model...',
    selectModelPlaceholder: 'Select Model...',
    selectedModel: 'Selected Model:',
    provider: 'Provider:',
    status: 'Status:',
    description: 'Description:',
    unknownModel: 'Unknown model',
    availableModels: 'Available Models:',
    // Sign out confirm
    signOutConfirm: 'Are you sure you want to sign out?',
    // Availability and badges
    badgeDefault: '(Default)',
    badgeUnavailable: '(Unavailable)',
    aiProviderFallback: 'AI',
    available: 'Available',
    unavailable: 'Unavailable',
    // Web search settings
    webSearchEnabled: 'Web Search',
    webSearchEnabledDescription: 'Enable AI-powered web search for real-time information',
  },
  ja: {
    appTitle: 'AI株式アシスタント',
    appSubtitle: 'あなたの賢い投資パートナー',
    authLogin: 'ログイン',
    authRegister: 'アカウント作成',
    settings: '設定',
    language: '言語',
    selectLanguage: '言語を選択',
    english: '英語',
    japanese: '日本語',
    signOut: 'サインアウト',
    cancel: 'キャンセル',
    closeSidebar: 'サイドバーを閉じる',
    openSidebar: 'サイドバーを開く',
    searchChats: 'チャットを検索…',
    noChatsFound: 'チャットが見つかりません',
    tryDifferentSearch: '別の検索語をお試しください',
    noChatsYet: 'まだチャットがありません',
    clickNewChat: '「新しいチャット」をクリックして会話を開始してください',
    modelSelection: 'AIモデル選択',
    systemPrompt: 'システムプロンプト',
    placeholder: '株式や市場データ、投資分析について質問してください…',
    // Onboarding/empty state
    onboardingTitle: 'AI株式アシスタントへようこそ',
    onboardingBody: '株式、市場データ、投資分析、投資戦略など何でも聞いてください。リアルタイム株価、過去データ、企業プロフィール、ニュース、リスク評価までサポートします。',
    usingModel: '使用モデル',
    // Tool activity panel
    toolActivity: 'ツールの実行状況',
    statusRunning: '実行中',
    statusCompleted: '完了',
    statusError: 'エラー',
    // Cached indicator
    cachedNotice: 'キャッシュから配信 — キャッシュ時はツールが実行されない場合があります。',
    // Errors
    mustSignIn: 'チャットを利用するにはサインインしてください。',
    loginFailed: 'ログインに失敗しました',
    registerFailed: '登録に失敗しました',
    streamingFailed: 'ストリーミングに失敗しました',
    stop: '停止',
    send: '送信',
    newChat: '新しいチャット',
    messagesCountSuffix: '件のメッセージ',
    responding: '応答中...',
    // Settings details
    selectAIModelPlaceholder: 'AIモデルを選択…',
    selectModelPlaceholder: 'モデルを選択…',
    selectedModel: '選択中のモデル:',
    provider: 'プロバイダー:',
    status: 'ステータス:',
    description: '説明:',
    unknownModel: '不明なモデル',
    availableModels: '利用可能なモデル:',
    // Sign out confirm
    signOutConfirm: 'サインアウトしてもよろしいですか？',
    // Availability and badges
    badgeDefault: '（デフォルト）',
    badgeUnavailable: '（利用不可）',
    aiProviderFallback: 'AI',
    available: '利用可能',
    unavailable: '利用不可',
    // Web search settings
    webSearchEnabled: 'ウェブ検索',
    webSearchEnabledDescription: 'リアルタイム情報のためのAI駆動ウェブ検索を有効にする',
  },
}

let cachedLocale: Locale | null = null

export function getStoredLocale(): Locale {
  if (cachedLocale !== null) return cachedLocale

  try {
    const v = localStorage.getItem(STORAGE_KEY)
    if (v === 'ja' || v === 'en') {
      cachedLocale = v
      return v
    }
  } catch {}

  cachedLocale = 'en'
  return 'en'
}

export function setStoredLocale(locale: Locale) {
  cachedLocale = locale
  translationCache.clear() // Clear cache when locale changes
  try { localStorage.setItem(STORAGE_KEY, locale) } catch {}
}

export function useI18n() {
  const stored = getStoredLocale()
  let current: Locale = stored

  // Memoize the translation function to avoid recreating it on every render
  const t = useCallback((key: keyof typeof dict['en']): string => {
    const cacheKey = `${current}:${key}`

    if (translationCache.has(cacheKey)) {
      return translationCache.get(cacheKey)!
    }

    const table = dict[current]
    const result = table[key] ?? key
    translationCache.set(cacheKey, result)
    return result
  }, [current])

  const setLocale = useCallback((l: Locale) => {
    current = l
    setStoredLocale(l)
  }, [])

  return { t, locale: current, setLocale }
}

// Optimized batch translation function for multiple keys
export function batchTranslate(keys: Array<keyof typeof dict['en']>, locale?: Locale): Record<string, string> {
  const l = locale || getStoredLocale()
  const result: Record<string, string> = {}

  for (const key of keys) {
    const cacheKey = `${l}:${key}`

    if (translationCache.has(cacheKey)) {
      result[key] = translationCache.get(cacheKey)!
    } else {
      const translation = dict[l][key] || dict.en[key] || key
      translationCache.set(cacheKey, translation)
      result[key] = translation
    }
  }

  return result
}

export function translate(key: keyof typeof dict['en'], locale?: Locale): string {
  const l = locale || getStoredLocale()
  const cacheKey = `${l}:${key}`

  // Check cache first
  if (translationCache.has(cacheKey)) {
    return translationCache.get(cacheKey)!
  }

  // Get translation and cache it
  const result = dict[l][key] || dict.en[key] || key
  translationCache.set(cacheKey, result)
  return result
}
