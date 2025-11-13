#!/usr/bin/env python3
"""
Debug why Japanese banking query returns no results.
Test different search strategies and API responses.
"""

import asyncio
import os
import sys
sys.path.append('/home/khaitran/PycharmProjects/Azure-OpenAI_StockTool')

from app.services.perplexity_web_search import PerplexityWebSearchService, BraveSearchClient
from ddgs import DDGS

async def debug_japanese_banking_search():
    """Debug the Japanese banking search to identify why no results are returned."""
    
    # Original query
    original_query = "八十二銀行と長野銀行の統合関連ニュースを検索してください。"
    
    # Test variations
    test_queries = [
        {
            "query": original_query,
            "description": "Original Japanese query (full sentence)"
        },
        {
            "query": "八十二銀行 長野銀行 統合",
            "description": "Simplified Japanese keywords only"
        },
        {
            "query": "Hachijuni Bank Nagano Bank merger",
            "description": "English translation"
        },
        {
            "query": "八十二銀行 統合 ニュース",
            "description": "Hachijuni Bank merger news (Japanese)"
        },
        {
            "query": "regional bank merger Japan 2024",
            "description": "English regional bank merger query"
        }
    ]
    
    print("🔍 Debugging Japanese Banking Search")
    print("=" * 60)
    
    # Test 1: Direct DDGS search (no wrapper)
    print("\n1️⃣ Testing Direct DDGS Search")
    print("-" * 40)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\nTest {i}: {description}")
        print(f"Query: {query}")
        
        try:
            # Direct DDGS test
            ddgs = DDGS()
            results = list(ddgs.text(
                keywords=query,
                region='jp-jp',  # Japan region
                safesearch='moderate',
                timelimit='y',  # Past year
                max_results=5
            ))
            
            print(f"   DDGS Results: {len(results)}")
            for j, result in enumerate(results[:3], 1):
                print(f"   {j}. {result.get('title', 'No title')[:60]}...")
                print(f"      URL: {result.get('href', 'No URL')}")
                
        except Exception as e:
            print(f"   ❌ DDGS Error: {str(e)}")
        
        # Small delay between queries
        await asyncio.sleep(1)
    
    # Test 2: Brave Search API test
    print(f"\n\n2️⃣ Testing Brave Search API")
    print("-" * 40)
    
    async with BraveSearchClient() as brave_client:
        if brave_client.is_available:
            for i, test_case in enumerate(test_queries[:3], 1):  # Test first 3 queries
                query = test_case["query"]
                description = test_case["description"]
                
                print(f"\nBrave Test {i}: {description}")
                print(f"Query: {query}")
                
                try:
                    results = await brave_client.search(
                        query=query,
                        count=5,
                        country='JP' if any(ord(c) > 127 for c in query) else 'ALL'
                    )
                    
                    print(f"   Brave Results: {len(results)}")
                    for j, result in enumerate(results[:3], 1):
                        print(f"   {j}. {result.get('title', 'No title')[:60]}...")
                        print(f"      URL: {result.get('url', 'No URL')}")
                        print(f"      Relevance: {result.get('relevance_score', 0):.3f}")
                        
                except Exception as e:
                    print(f"   ❌ Brave Error: {str(e)}")
                
                await asyncio.sleep(2)  # Respect rate limits
        else:
            print("   ❌ Brave Search API not available")
    
    # Test 3: Full PerplexityWebSearchService test
    print(f"\n\n3️⃣ Testing Full Search Service")
    print("-" * 40)
    
    async with PerplexityWebSearchService() as search_service:
        for i, test_case in enumerate(test_queries[:2], 1):  # Test first 2 queries
            query = test_case["query"]
            description = test_case["description"]
            
            print(f"\nFull Test {i}: {description}")
            print(f"Query: {query}")
            
            try:
                response = await search_service.perplexity_search(
                    query=query,
                    max_results=5,
                    synthesize_answer=False,  # Skip synthesis for debugging
                    include_recent=True
                )
                
                print(f"   Service Results: {len(response.sources)}")
                print(f"   Search Time: {response.search_time:.2f}s")
                
                for j, source in enumerate(response.sources[:3], 1):
                    print(f"   {j}. {source.title[:60]}...")
                    print(f"      URL: {source.url}")
                    print(f"      Combined Score: {source.combined_score:.3f}")
                    print(f"      Domain Boost: {source.domain_boost:.2f}x")
                    
            except Exception as e:
                print(f"   ❌ Service Error: {str(e)}")
                import traceback
                traceback.print_exc()
            
            await asyncio.sleep(2)
    
    # Test 4: Check language detection
    print(f"\n\n4️⃣ Language Detection Analysis")
    print("-" * 40)
    
    for test_case in test_queries:
        query = test_case["query"]
        description = test_case["description"]
        
        # Check character encoding
        has_japanese = any(ord(c) > 127 for c in query)
        japanese_chars = [c for c in query if ord(c) > 127]
        
        print(f"\nQuery: {query}")
        print(f"  Description: {description}")
        print(f"  Has Japanese chars: {has_japanese}")
        print(f"  Japanese chars: {''.join(japanese_chars[:10])}")
        print(f"  Length: {len(query)} chars")
    
    print(f"\n\n📋 Analysis Summary:")
    print(f"=" * 50)
    print(f"Possible reasons for no results:")
    print(f"1. 🌐 Regional restrictions - Japanese content may not be in search APIs")
    print(f"2. 🔍 Query complexity - Full sentences vs keywords")
    print(f"3. 📅 Recency - These banks may not have recent merger news")
    print(f"4. 🔑 API limitations - Free tiers may not cover Japanese content well")
    print(f"5. 🎯 Domain filtering - Japanese news sites may not be in domain priors")
    print(f"6. 🚫 Rate limiting - Search APIs may be throttling requests")

if __name__ == "__main__":
    asyncio.run(debug_japanese_banking_search())