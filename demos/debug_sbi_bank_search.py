#!/usr/bin/env python3
"""
Debug why SBI Sumishin Net Bank search returns no results.
Test different search strategies and API responses.
"""

import asyncio
import os
import sys
sys.path.append('/home/khaitran/PycharmProjects/Azure-OpenAI_StockTool')

from app.services.perplexity_web_search import PerplexityWebSearchService, BraveSearchClient
from app.services.perplexity_web_search import get_openai_client
from ddgs import DDGS

async def debug_sbi_bank_search():
    """Debug the SBI Sumishin Net Bank search to identify why no results are returned."""
    
    # Original query
    original_query = "SBI住信ネット銀行の顧客基盤と成長戦略は？"
    
    # Test variations
    test_queries = [
        {
            "query": original_query,
            "description": "Original Japanese query (full question)"
        },
        {
            "query": "SBI住信ネット銀行 顧客基盤 成長戦略",
            "description": "Simplified Japanese keywords only"
        },
        {
            "query": "SBI住信ネット銀行 経営戦略",
            "description": "Simplified to SBI Sumishin Net Bank management strategy"
        },
        {
            "query": "SBI Sumishin Net Bank customer base growth strategy",
            "description": "English translation"
        },
        {
            "query": "SBI住信ネット銀行 戦略 2024",
            "description": "SBI Bank strategy 2024 (Japanese)"
        }
    ]
    
    print("🔍 Debugging SBI Sumishin Net Bank Search")
    print("=" * 60)
    
    # First let's check how the query enhancer transforms our query
    print("\n0️⃣ Testing Query Enhancement")
    print("-" * 40)
    
    service = PerplexityWebSearchService()
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\nTest {i}: {description}")
        print(f"Query: {query}")
        
        # Test LLM query enhancement
        try:
            # Properly await the async response
            enhanced_query = await service._llm_synthesize_query(query, include_recent=True)
            print(f"   ✅ LLM Enhanced: '{enhanced_query}'")
        except Exception as e:
            print(f"   ❌ LLM Enhancement Failed: {str(e)}")
            
            # Try fallback enhancement
            fallback_query = service._fallback_enhance_query(query, include_recent=True)
            print(f"   ⚠️ Fallback Enhanced: '{fallback_query}'")
        
        # Test Japanese simplification
        simplified = service._simplify_japanese_query(query)
        print(f"   🔄 Simplified: '{simplified}'")
        
        # Small delay
        await asyncio.sleep(1)
    
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
            for i, test_case in enumerate(test_queries, 1):
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
        for i, test_case in enumerate(test_queries, 1):
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
                    
            except Exception as e:
                print(f"   ❌ Service Error: {str(e)}")
                import traceback
                traceback.print_exc()
            
            await asyncio.sleep(2)
    
    # Test 4: Check region selection logic
    print(f"\n\n4️⃣ Region Selection Analysis")
    print("-" * 40)
    
    for test_case in test_queries:
        query = test_case["query"]
        description = test_case["description"]
        
        # Check region selection
        region = service._get_optimal_ddgs_region(query)
        
        print(f"\nQuery: {query}")
        print(f"  Description: {description}")
        print(f"  Selected DDGS region: {region}")
    
    print(f"\n\n📋 Analysis Summary and Recommendations:")
    print(f"=" * 50)
    print(f"Possible fixes for SBI Sumishin Net Bank search:")
    print(f"1. 🔤 Query simplification: Replace '？' with simpler keywords")
    print(f"2. 🌐 Force JP region explicitly: Use 'jp-jp' for all queries")
    print(f"3. 🔍 Try alternative term: Change from '顧客基盤と成長戦略' to '経営戦略'")
    print(f"4. 🎯 Add domain priors: Add financial sites to domain priority list")
    print(f"5. 🔄 Manual fallback: If automated search fails, try English translation")
    print(f"6. 🧠 Improve query enhancement: Optimize LLM prompt for Japanese financial terms")

if __name__ == "__main__":
    asyncio.run(debug_sbi_bank_search())