/**
 * Settings modal component - reusable across the app
 */

import React from 'react'
import { Modal, Textarea } from '../ui'
import { translate, type Locale } from '../../i18n'
import type { ModelsResponse, AppSettings } from '../../types'

export interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  settings: AppSettings
  onSettingsChange: (updater: (prev: AppSettings) => AppSettings) => void
  models: ModelsResponse | null
  locale: Locale
}

const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onSettingsChange,
  models,
  locale
}) => {
  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={translate('settings', locale)}
    >
      <div className="space-y-6">
        <Textarea
          label={translate('systemPrompt', locale)}
          value={settings.systemPrompt}
          onChange={(e) => onSettingsChange(prev => ({ 
            ...prev, 
            systemPrompt: e.target.value 
          }))}
          rows={6}
          placeholder="Enter system prompt..."
        />

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            {translate('language', locale)}
          </label>
          <select
            value={settings.locale}
            onChange={(e) => onSettingsChange(prev => ({ 
              ...prev, 
              locale: e.target.value as Locale 
            }))}
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
              value={settings.deployment}
              onChange={(e) => onSettingsChange(prev => ({ 
                ...prev, 
                deployment: e.target.value 
              }))}
              className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="">{translate('selectModelPlaceholder', locale)}</option>
              {models.available.map((model) => (
                <option key={model.id} value={model.id} disabled={!model.available}>
                  {model.name} {model.default ? '(Default)' : ''} - {model.provider.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Web Search Toggle */}
        <div className="flex items-center justify-between">
          <div>
            <label className="block text-sm font-medium text-gray-300">
              {translate('webSearchEnabled', locale)}
            </label>
            <p className="text-xs text-gray-500 mt-1">
              {translate('webSearchEnabledDescription', locale)}
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={settings.webSearchEnabled}
            onClick={() => onSettingsChange(prev => ({
              ...prev,
              webSearchEnabled: !prev.webSearchEnabled
            }))}
            className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 ${
              settings.webSearchEnabled ? 'bg-blue-600' : 'bg-gray-600'
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                settings.webSearchEnabled ? 'translate-x-5' : 'translate-x-0'
              }`}
            />
          </button>
        </div>
      </div>
    </Modal>
  )
}

export default SettingsModal