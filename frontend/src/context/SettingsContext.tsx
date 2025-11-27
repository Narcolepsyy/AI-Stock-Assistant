import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { listModels, type ModelsResponse } from '../lib/api'
import { DEFAULT_SYSTEM_PROMPT, STORAGE_KEYS } from '../constants'
import { getStoredLocale, setStoredLocale, type Locale } from '../i18n'
import type { AppSettings } from '../types'

type ModelsStatus = 'idle' | 'loading' | 'success' | 'error'

export interface SettingsContextValue {
  settings: AppSettings
  models: ModelsResponse | null
  modelsStatus: ModelsStatus
  modelsError: string | null
  refreshModels: () => Promise<void>
  updateSettings: (updater: (prev: AppSettings) => AppSettings) => void
  setSystemPrompt: (prompt: string) => void
  setLocale: (locale: Locale) => void
  setDeployment: (deployment: string) => void
}

const SettingsContext = createContext<SettingsContextValue | undefined>(undefined)

// Hook must be in the same file to access the context
export function useSettings(): SettingsContextValue {
  const context = useContext(SettingsContext)
  if (!context) {
    throw new Error('useSettings must be used within a SettingsProvider')
  }
  return context
}

const loadStoredSettings = (): Partial<AppSettings> => {
  try {
    const storedPrompt = localStorage.getItem(STORAGE_KEYS.SYSTEM_PROMPT)
    const storedDeployment = localStorage.getItem(STORAGE_KEYS.SELECTED_MODEL) || ''
    const storedWebSearch = localStorage.getItem(STORAGE_KEYS.WEB_SEARCH_ENABLED)
    const locale = getStoredLocale()
    return {
      systemPrompt: storedPrompt ?? DEFAULT_SYSTEM_PROMPT,
      deployment: storedDeployment,
      locale,
      webSearchEnabled: storedWebSearch !== null ? storedWebSearch === 'true' : true,
    }
  } catch {
    return {
      systemPrompt: DEFAULT_SYSTEM_PROMPT,
      deployment: '',
      locale: getStoredLocale(),
      webSearchEnabled: true,
    }
  }
}

const ensureDeployment = (current: AppSettings, models: ModelsResponse | null): AppSettings => {
  if (!models) {
    return current
  }

  const availableIds = new Set(models.available.map((model) => model.id))
  if (current.deployment && availableIds.has(current.deployment)) {
    return current
  }

  const fallback = models.default || models.available[0]?.id || ''
  return fallback && fallback !== current.deployment
    ? { ...current, deployment: fallback }
    : current
}

export const SettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [settings, setSettings] = useState<AppSettings>(() => ({
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
    deployment: '',
    locale: getStoredLocale(),
    webSearchEnabled: true,
    ...loadStoredSettings(),
  }))

  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [modelsStatus, setModelsStatus] = useState<ModelsStatus>('idle')
  const [modelsError, setModelsError] = useState<string | null>(null)

  const persistSettings = useCallback((next: AppSettings) => {
    try {
      localStorage.setItem(STORAGE_KEYS.SYSTEM_PROMPT, next.systemPrompt)
      if (next.deployment) {
        localStorage.setItem(STORAGE_KEYS.SELECTED_MODEL, next.deployment)
      } else {
        localStorage.removeItem(STORAGE_KEYS.SELECTED_MODEL)
      }
      localStorage.setItem(STORAGE_KEYS.WEB_SEARCH_ENABLED, String(next.webSearchEnabled))
      setStoredLocale(next.locale)
    } catch {
      // Swallow storage errors silently
    }
  }, [])

  const updateSettings = useCallback(
    (updater: (prev: AppSettings) => AppSettings) => {
      setSettings((prev) => {
        const next = updater(prev)
        persistSettings(next)
        return next
      })
    },
    [persistSettings]
  )

  const setSystemPrompt = useCallback(
    (prompt: string) => {
      updateSettings((prev) => ({ ...prev, systemPrompt: prompt }))
    },
    [updateSettings]
  )

  const setLocale = useCallback(
    (locale: Locale) => {
      updateSettings((prev) => ({ ...prev, locale }))
    },
    [updateSettings]
  )

  const setDeployment = useCallback(
    (deployment: string) => {
      updateSettings((prev) => ({ ...prev, deployment }))
    },
    [updateSettings]
  )

  const refreshModels = useCallback(async () => {
    setModelsStatus('loading')
    setModelsError(null)

    try {
      const fetched = await listModels()
      setModels(fetched)
      setModelsStatus('success')

      setSettings((prev) => {
        const next = ensureDeployment(prev, fetched)
        if (next !== prev) {
          persistSettings(next)
        }
        return next
      })
    } catch (error: any) {
      const message = error?.message || 'Failed to load models'
      setModelsStatus('error')
      setModelsError(message)
    }
  }, [persistSettings])

  useEffect(() => {
    refreshModels()
  }, [refreshModels])

  const contextValue = useMemo<SettingsContextValue>(() => ({
    settings,
    models,
    modelsStatus,
    modelsError,
    refreshModels,
    updateSettings,
    setSystemPrompt,
    setLocale,
    setDeployment,
  }), [settings, models, modelsStatus, modelsError, refreshModels, updateSettings, setSystemPrompt, setLocale, setDeployment])

  return <SettingsContext.Provider value={contextValue}>{children}</SettingsContext.Provider>
}
