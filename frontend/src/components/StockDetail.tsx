import React, { useState, useEffect } from 'react'
import { getToken, apiBase } from '../lib/api'
import PriceQuoteCard, { QuoteData } from './PriceQuoteCard'
import { LoadingSpinner } from './ui'

interface CompanyProfile {
  sector?: string
  industry?: string
  website?: string
  description?: string
  country?: string
  city?: string
  employees?: number
  phone?: string
  address?: string
}

interface NewsArticle {
  title: string
  publisher: string
  link: string
  published: string
  thumbnail?: string
}

interface StockDetailProps {
  symbol: string
  onBack: () => void
}

const StockDetail: React.FC<StockDetailProps> = ({ symbol, onBack }) => {
  const [quoteData, setQuoteData] = useState<QuoteData | null>(null)
  const [companyProfile, setCompanyProfile] = useState<CompanyProfile | null>(null)
  const [newsArticles, setNewsArticles] = useState<NewsArticle[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true
    const pollInterval = 5000 // 5 seconds

    const fetchStockData = async (isBackground = false) => {
      if (!isBackground) {
        setIsLoading(true)
        setError(null)
      }

      try {
        const token = getToken()
        if (!token) {
          if (isMounted) setError('Authentication required')
          return
        }

        const response = await fetch(`${apiBase()}/dashboard/jp/quote/${symbol}?with_chart=true`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (!response.ok) {
          throw new Error(`Failed to fetch stock data: ${response.statusText}`)
        }

        const data = await response.json()

        if (!isMounted) return

        // Transform the data to match QuoteData interface
        const transformedData: QuoteData = {
          symbol: data.symbol || symbol,
          price: data.current_price,
          currency: data.currency || 'JPY',
          change: data.change,
          change_percent: data.percent_change,
          previous_close: data.previous_close,
          day_open: data.open,
          day_high: data.day_high,
          day_low: data.day_low,
          volume: data.volume,
          market_cap: data.market_cap,
          shares_outstanding: data.shares_outstanding,
          year_high: data.year_high,
          year_low: data.year_low,
          eps: data.eps,
          pe_ratio: data.pe_ratio,
          as_of: data.datetime || data.timestamp,
          chart: data.chart
        }

        setQuoteData(transformedData)

        // Store company profile separately if available
        if (data.company_profile) {
          setCompanyProfile(data.company_profile)
        }

        // Store news articles if available
        if (data.news && Array.isArray(data.news)) {
          setNewsArticles(data.news)
        }
      } catch (err) {
        console.error('Error fetching stock data:', err)
        if (isMounted && !isBackground) {
          setError(err instanceof Error ? err.message : 'Failed to load stock data')
        }
      } finally {
        if (isMounted && !isBackground) {
          setIsLoading(false)
        }
      }
    }

    // Initial fetch
    fetchStockData()

    // Set up polling
    const intervalId = setInterval(() => {
      fetchStockData(true)
    }, pollInterval)

    return () => {
      isMounted = false
      clearInterval(intervalId)
    }
  }, [symbol])

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <LoadingSpinner size="lg" label={`Loading ${symbol}...`} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 p-6">
        <div className="max-w-7xl mx-auto">
          <button
            onClick={onBack}
            className="mb-4 flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Dashboard
          </button>

          <div className="bg-red-900/20 border border-red-700 rounded-lg p-6 text-center">
            <svg className="w-12 h-12 text-red-400 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h2 className="text-xl font-bold text-red-400 mb-2">Error Loading Stock Data</h2>
            <p className="text-gray-300">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  if (!quoteData) {
    return (
      <div className="min-h-screen bg-gray-900 p-6">
        <div className="max-w-7xl mx-auto">
          <button
            onClick={onBack}
            className="mb-4 flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Dashboard
          </button>

          <div className="text-center text-gray-400 py-12">
            <p>No data available for {symbol}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen bg-gray-900 overflow-hidden flex flex-col">
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto p-6 pb-12">
          {/* Back Navigation */}
          <button
            onClick={onBack}
            className="mb-6 flex items-center gap-2 text-gray-400 hover:text-white transition-colors group"
          >
            <svg className="w-5 h-5 group-hover:-translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Dashboard
          </button>

          {/* Stock Header */}
          <div className="mb-6 flex justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">
                {quoteData.symbol}
              </h1>
              <p className="text-gray-400">Tokyo Stock Exchange</p>
            </div>
            <div className="flex items-center gap-2 px-3 py-1 bg-green-900/30 border border-green-800 rounded-full">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              <span className="text-xs font-medium text-green-400">Live Updates</span>
            </div>
          </div>

          {/* Price Quote Card with Chart */}
          <PriceQuoteCard data={quoteData} />

          {/* Additional Information Sections */}
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Company Info */}
            <div className="bg-gradient-to-br from-gray-900/70 via-gray-900/30 to-gray-800/40 border border-gray-700/40 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Company Information</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Symbol</span>
                  <span className="text-white font-medium">{quoteData.symbol}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Currency</span>
                  <span className="text-white font-medium">{quoteData.currency}</span>
                </div>
                {companyProfile?.sector && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Sector</span>
                    <span className="text-white font-medium">{companyProfile.sector}</span>
                  </div>
                )}
                {companyProfile?.industry && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Industry</span>
                    <span className="text-white font-medium">{companyProfile.industry}</span>
                  </div>
                )}
                {companyProfile?.employees && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Employees</span>
                    <span className="text-white font-medium">
                      {companyProfile.employees.toLocaleString()}
                    </span>
                  </div>
                )}
                {quoteData.shares_outstanding && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Shares Outstanding</span>
                    <span className="text-white font-medium">
                      {new Intl.NumberFormat('en-US', {
                        notation: 'compact',
                        maximumFractionDigits: 2
                      }).format(quoteData.shares_outstanding)}
                    </span>
                  </div>
                )}
                {companyProfile?.phone && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Phone</span>
                    <a
                      href={`tel:${companyProfile.phone}`}
                      className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1"
                    >
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                      </svg>
                      {companyProfile.phone}
                    </a>
                  </div>
                )}
                {companyProfile?.website && (
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Website</span>
                    <a
                      href={companyProfile.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 text-sm flex items-center gap-1"
                    >
                      Visit
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </a>
                  </div>
                )}
              </div>
            </div>

            {/* Market Statistics */}
            <div className="bg-gradient-to-br from-gray-900/70 via-gray-900/30 to-gray-800/40 border border-gray-700/40 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">Market Statistics</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">52-Week Range</span>
                  <span className="text-white font-medium">
                    {quoteData.year_low ? `¥${quoteData.year_low.toFixed(2)}` : '—'} - {quoteData.year_high ? `¥${quoteData.year_high.toFixed(2)}` : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Today's Range</span>
                  <span className="text-white font-medium">
                    {quoteData.day_low ? `¥${quoteData.day_low.toFixed(2)}` : '—'} - {quoteData.day_high ? `¥${quoteData.day_high.toFixed(2)}` : '—'}
                  </span>
                </div>
                {quoteData.as_of && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Last Updated</span>
                    <span className="text-white font-medium text-xs">
                      {new Date(quoteData.as_of).toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Company Description */}
          {companyProfile?.description && (
            <div className="mt-6 bg-gradient-to-br from-gray-900/70 via-gray-900/30 to-gray-800/40 border border-gray-700/40 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">About</h2>
              <p className="text-gray-300 leading-relaxed">{companyProfile.description}</p>
              {(companyProfile.country || companyProfile.city || companyProfile.address) && (
                <div className="mt-4 pt-4 border-t border-gray-700/40 flex items-start gap-2 text-sm text-gray-400">
                  <svg className="w-4 h-4 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <div>
                    {companyProfile.address && <div>{companyProfile.address}</div>}
                    {[companyProfile.city, companyProfile.country].filter(Boolean).join(', ')}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* News Section */}
          {newsArticles.length > 0 && (
            <div className="mt-6 bg-gradient-to-br from-gray-900/70 via-gray-900/30 to-gray-800/40 border border-gray-700/40 rounded-xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
                </svg>
                Latest News
              </h2>
              <div className="space-y-4">
                {newsArticles.map((article, idx) => (
                  <a
                    key={idx}
                    href={article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block group hover:bg-gray-800/30 rounded-lg p-4 transition-colors border border-transparent hover:border-gray-700/40"
                  >
                    <div className="flex gap-4">
                      {article.thumbnail && (
                        <div className="flex-shrink-0">
                          <img
                            src={article.thumbnail}
                            alt={article.title}
                            className="w-24 h-24 object-cover rounded-lg"
                            onError={(e) => {
                              // Hide image if it fails to load
                              e.currentTarget.style.display = 'none'
                            }}
                          />
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <h3 className="font-medium text-white group-hover:text-blue-400 transition-colors mb-2 line-clamp-2">
                          {article.title}
                        </h3>
                        <div className="flex items-center gap-3 text-xs text-gray-400">
                          <span className="flex items-center gap-1">
                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-6-3a2 2 0 11-4 0 2 2 0 014 0zm-2 4a5 5 0 00-4.546 2.916A5.986 5.986 0 0010 16a5.986 5.986 0 004.546-2.084A5 5 0 0010 11z" clipRule="evenodd" />
                            </svg>
                            {article.publisher}
                          </span>
                          <span>•</span>
                          <span className="flex items-center gap-1">
                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                            {new Date(article.published).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                      <svg className="w-5 h-5 text-gray-500 group-hover:text-blue-400 transition-colors flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                      </svg>
                    </div>
                  </a>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default StockDetail
