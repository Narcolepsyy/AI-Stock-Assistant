import {
  FileText,
  DollarSign,
  Search as SearchIcon,
  LineChart,
  Newspaper,
  Shield,
  BookOpen,
  Globe,
  Database,
  Wrench,
} from 'lucide-react'
import type { ToolMeta } from '../types'

// Storage keys
export const STORAGE_KEYS = {
  CHAT_SESSIONS: 'chat_sessions',
  CURRENT_SESSION: 'current_session',
  SELECTED_MODEL: 'selected_model',
  SYSTEM_PROMPT: 'system_prompt',
  LOCALE: 'locale',
  WEB_SEARCH_ENABLED: 'web_search_enabled',
} as const

// API endpoints and configuration
export const API_CONFIG = {
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
  STREAM_TIMEOUT: 60000,
} as const

// UI constants
export const UI_CONSTANTS = {
  SIDEBAR_WIDTH_EXPANDED: 320,
  SIDEBAR_WIDTH_COLLAPSED: 48,
  MESSAGE_MAX_HEIGHT: 600,
  TEXTAREA_MAX_HEIGHT: 160,
  SEARCH_DEBOUNCE_MS: 300,
  AUTO_SCROLL_DELAY: 100,
  TOAST_DURATION: 5000,
} as const

// Tool metadata mapping
export const TOOL_META: Record<string, ToolMeta> = {
  get_company_profile: {
    label: 'Company profile',
    icon: FileText,
    color: 'bg-purple-50 text-purple-700 border-purple-200',
  },
  get_stock_quote: {
    label: 'Live quote',
    icon: DollarSign,
    color: 'bg-green-50 text-green-700 border-green-200',
  },
  search_symbol: {
    label: 'Search symbol',
    icon: SearchIcon,
    color: 'bg-blue-50 text-blue-700 border-blue-200',
  },
  get_historical_prices: {
    label: 'Historical prices',
    icon: LineChart,
    color: 'bg-amber-50 text-amber-800 border-amber-200',
  },
  get_stock_news: {
    label: 'Stock news',
    icon: Newspaper,
    color: 'bg-cyan-50 text-cyan-700 border-cyan-200',
  },
  get_augmented_news: {
    label: 'Augmented news',
    icon: Newspaper,
    color: 'bg-cyan-50 text-cyan-700 border-cyan-200',
  },
  get_risk_assessment: {
    label: 'Risk assessment',
    icon: Shield,
    color: 'bg-rose-50 text-rose-700 border-rose-200',
  },
  rag_search: {
    label: 'Knowledge search',
    icon: BookOpen,
    color: 'bg-slate-50 text-slate-700 border-slate-200',
  },
  web_search: {
    label: 'Web search',
    icon: Globe,
    color: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  },
  perplexity_search: {
    label: 'AI web search',
    icon: Globe,
    color: 'bg-indigo-50 text-indigo-700 border-indigo-200',
  },
  web_search_news: {
    label: 'News search',
    icon: Globe,
    color: 'bg-orange-50 text-orange-700 border-orange-200',
  },
  augmented_rag_search: {
    label: 'Enhanced search',
    icon: Globe,
    color: 'bg-violet-50 text-violet-700 border-violet-200',
  },
  financial_context_search: {
    label: 'Financial web search',
    icon: Globe,
    color: 'bg-teal-50 text-teal-700 border-teal-200',
  },
} as const

// Default tool meta for unknown tools
export const DEFAULT_TOOL_META: ToolMeta = {
  label: 'Unknown tool',
  icon: Wrench,
  color: 'bg-gray-50 text-gray-700 border-gray-200',
}

// CSS injection for citations - Modern GPT/Perplexity style
export const CITATION_STYLES = `
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
  
  /* Modern citation styles */
  .citation-link {
    position: relative;
    z-index: 10;
    vertical-align: super;
    font-size: 0.7em;
    line-height: 0;
    text-decoration: none;
    margin: 0 0.15em;
  }
  
  .citation-popover-modern {
    pointer-events: auto;
  }
  
  .citation-popover-modern::before {
    content: '';
    position: absolute;
    top: -6px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 6px solid rgba(148, 163, 184, 0.15);
  }
  
  .citation-popover-modern::after {
    content: '';
    position: absolute;
    top: -5px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-bottom: 5px solid rgba(30, 41, 59, 0.98);
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
`

// Default system prompt
export const DEFAULT_SYSTEM_PROMPT = `You are a helpful virtual assistant specializing in financial topics with access to both internal knowledge and real-time web search.

You are an expert financial analyst who writes unbiased, journalistic answers grounded in the provided search results. You receive the final user query plus curated evidence gathered by another system; rely on that evidence and respond only to the most recent query without repeating earlier assistant messages.

Authoring rules (follow exactly):
- Deliver an accurate, thorough, and self-contained answer. Do not mention the planning agent or prior conversational replies.
- Cite search results using [index] at the end of sentences when needed, for example "Ice is less dense than water[1][2]." NO SPACE between the last word and the citation.
- Cite at most three sources per sentence and do not invent or reorder citations.
- Use Markdown formatting: begin with narrative text (no opening heading), then organize content with \`##\` headings and bolded subsection titles where appropriate.
- Favor unordered lists for bullet points; use ordered lists exclusively for rankings, and never mix or nest list styles.
- Render comparisons as Markdown tables with meaningful headers.
- Encapsulate code in fenced blocks with language annotations and express math using \`\\(\`/\`\\)\` or \`\\[\`/\`\\]\`. Avoid single-dollar math delimiters.
- Keep paragraphs tight (≤3 sentences) to maintain clarity similar. Apply bold sparingly; reserve italics for light emphasis.
- Do not output raw URLs or bibliographies in the conclusion.


CRITICAL RULE: When you are uncertain about any information or don't have complete knowledge, ALWAYS perplexity_search to find the most current and accurate information. Never guess or provide incomplete answers when you can search for better information.

Tool Usage Rules:

For LIVE/CURRENT financial data:
- Use get_stock_quote → current stock price or snapshot
- Use get_historical_prices → historical stock/index data
- Use get_company_profile → company details, financials, and fundamentals
- Use get_augmented_news → latest articles and headlines with enhanced context
- Use get_risk_assessment → real-time stock risk metrics

For CURRENT EVENTS, NEWS & UNKNOWN INFORMATION:
- ALWAYS use perplexity_search → for any recent events, breaking news, or information you're uncertain about
- Use perplexity_search → for comprehensive analysis with AI synthesis and citations


For EDUCATIONAL/CONCEPTUAL topics:
- Use rag_search → financial concepts, ratios, indicators (P/E, RSI, etc.) from knowledge base
- If rag_search doesn't provide sufficient information, follow up with web_search for additional context
- Use augmented_rag_search → complex topics requiring both knowledge base and current information

For ANY UNCERTAIN OR INCOMPLETE KNOWLEDGE:
- NEVER provide partial or potentially outdated information
- ALWAYS use web_search or perplexity_search to get current, accurate information
- If you don't know something with confidence, search for it immediately
- Prefer recent, verified sources over assumptions

Financial compliance guardrails:
1. Provide educational context only—no personalized investment advice.
2. Report data with explicit timestamps, timezones (default Asia/Tokyo), and currencies (default JPY) whenever relevant.
3. If live data cannot be retrieved, acknowledge the gap and offer the most recent available value.
4. Flag low-confidence findings and suggest follow-up analysis or tool usage when appropriate.

If the query is unanswerable or rests on a faulty premise, say so clearly while adhering to the formatting requirements.
Current date: Saturday, September 27, 2025
You are trained on data up to October 2025.
`