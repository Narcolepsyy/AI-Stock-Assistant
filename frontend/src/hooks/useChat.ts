/**
 * Custom hook for managing chat streaming and API calls
 */

import { useState, useCallback, useRef } from 'react'
import { flushSync } from 'react-dom'
import { chatStream, cancelChatStream, type ChatRequest as ApiChatRequest } from '../lib/api'
import type { Message, LiveTool } from '../types'
import { generateId } from '../utils'
import { addLiveTools, updateLiveTools } from '../utils/tools'

interface ChatRequest {
  prompt: string
  system_prompt: string
  deployment: string
  conversation_id?: string
  locale: 'en' | 'ja'
  web_search_enabled?: boolean
}

export const useChat = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTools, setActiveTools] = useState<LiveTool[]>([])
  const [isCachedStream, setIsCachedStream] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  
  const accumulatedContentRef = useRef('')

  const sendMessage = useCallback(async (
    request: ChatRequest,
    onMessageUpdate: (updater: (prev: Message[]) => Message[]) => void
  ) => {
    setError(null)
    setIsLoading(true)
    setIsStreaming(true)
    setIsTyping(true)
    setActiveTools([])
    accumulatedContentRef.current = ''

    // Create streaming message placeholder
    const streamingMessage: Message = {
      id: generateId('assistant'),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    }

    // Add placeholder to messages
    onMessageUpdate(prev => [...prev, streamingMessage])

    try {
      await chatStream(
        request as ApiChatRequest,
        // onChunk handler
        (chunk) => {
          if (chunk.type === 'start') {
            if (chunk.conversation_id) {
              setConversationId(chunk.conversation_id)
            }
            if (chunk.cached === true) {
              setIsCachedStream(true)
            }
          } else if (chunk.type === 'content' && chunk.delta) {
            accumulatedContentRef.current += chunk.delta
            
            // Force synchronous update to prevent batching and show streaming effect
            flushSync(() => {
              // Update streaming message with accumulated content
              onMessageUpdate(prev => {
                if (prev.length === 0) return prev
                const lastIndex = prev.length - 1
                const lastMsg = prev[lastIndex]
                if (!lastMsg || lastMsg.role !== 'assistant') return prev
                
                const updatedMsg: Message = { 
                  ...lastMsg, 
                  content: accumulatedContentRef.current 
                }
                return [...prev.slice(0, lastIndex), updatedMsg]
              })
            })
          } else if (chunk.type === 'tool_calls') {
            const names: string[] = Array.isArray(chunk.tools) ? chunk.tools : []
            setActiveTools(prev => addLiveTools(prev, names))
            setIsTyping(true)
          } else if (chunk.type === 'tool_call') {
            const name = chunk.name as string | undefined
            const status = (chunk.status as 'completed' | 'error' | 'running' | undefined) || 'running'
            
            if (name) {
              setActiveTools(prev => updateLiveTools(prev, name, status, chunk.error))
            }
          }
        },
        // onComplete handler
        (data) => {
          const finalMessage: Message = {
            ...streamingMessage,
            content: accumulatedContentRef.current,
            toolCalls: data.tool_calls || [],
            timestamp: new Date(),
          }

          // Replace streaming placeholder with final message
          // Find the placeholder by ID to ensure we replace the correct message
          onMessageUpdate(prev => {
            const placeholderIndex = prev.findIndex(m => m.id === streamingMessage.id)
            if (placeholderIndex === -1) {
              // Placeholder not found, append final message
              return [...prev, finalMessage]
            }
            // Replace placeholder at its position
            const newMessages = [...prev]
            newMessages[placeholderIndex] = finalMessage
            return newMessages
          })

          if (data.conversation_id) {
            setConversationId(data.conversation_id)
          }

          // Reset states
          setIsStreaming(false)
          setIsTyping(false)
          setIsLoading(false)
          setTimeout(() => setActiveTools([]), 2000)
          setIsCachedStream(false)
        },
        // onError handler
        (error) => {
          setError(error)
          setIsStreaming(false)
          setIsTyping(false)
          setIsLoading(false)
          setTimeout(() => setActiveTools([]), 3000)
          setIsCachedStream(false)
          
          // Remove placeholder message on error
          onMessageUpdate(prev => {
            if (prev.length > 0 && 
                prev[prev.length - 1]?.role === 'assistant' && 
                prev[prev.length - 1]?.content === '') {
              return prev.slice(0, -1)
            }
            return prev
          })
        }
      )
    } catch (err: any) {
      const errorMessage = err?.message || 'Streaming failed'
      setError(errorMessage)
      setIsStreaming(false)
      setIsTyping(false)
      setIsLoading(false)
      setTimeout(() => setActiveTools([]), 3000)
      setIsCachedStream(false)
      
      // Remove placeholder message on error
      onMessageUpdate(prev => {
        if (prev.length > 0 && 
            prev[prev.length - 1]?.role === 'assistant' && 
            prev[prev.length - 1]?.content === '') {
          return prev.slice(0, -1)
        }
        return prev
      })
    }
  }, [])

  const stopGeneration = useCallback(() => {
    cancelChatStream()
    setIsStreaming(false)
    setIsTyping(false)
    setIsLoading(false)
    setTimeout(() => setActiveTools([]), 1000)
  }, [])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  return {
    isLoading,
    isStreaming,
    isTyping,
    error,
    activeTools,
    isCachedStream,
    conversationId,
    sendMessage,
    stopGeneration,
    clearError,
  }
}