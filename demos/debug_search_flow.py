#!/usr/bin/env python3
"""
Debug the full search flow to identify where results are being lost.
"""

import asyncio
import sys
sys.path.append('/home/khaitran/PycharmProjects/Azure-OpenAI_StockTool')

from app.services.perplexity_web_search import PerplexityWebSearchService

async def debug_search_flow():
    """Debug the search flow step by step."""
    
    query = "八十二銀行と長野銀行の統合関連ニュースを検索してください。"
    
    print("🔍 Debugging Search Flow")
    print("=" * 60)
    print(f"Original Query: {query}")
    
    async with PerplexityWebSearchService() as service:
        # Step 1: Test query enhancement
        enhanced_query = service._enhance_search_query(query, include_recent=True)
        print(f"Enhanced Query: {enhanced_query}")
        
        # Step 2: Test direct DDGS search
        print(f"\n📡 Testing _direct_ddgs_search...")
        try:
            ddgs_results = await service._direct_ddgs_search(enhanced_query, max_results=8)
            print(f"DDGS Raw Results: {len(ddgs_results)}")
            
            for i, result in enumerate(ddgs_results[:3], 1):
                print(f"  {i}. {result.get('title', 'No title')[:60]}...")
                print(f"     URL: {result.get('url', 'No URL')}")
                print(f"     Snippet: {result.get('snippet', 'No snippet')[:80]}...")
                
        except Exception as e:
            print(f"❌ DDGS Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 3: Test enhanced web search
        print(f"\n🌐 Testing _enhanced_web_search...")
        try:
            search_results = await service._enhanced_web_search(
                query=query,
                max_results=8,
                include_recent=True,
                time_limit=None
            )
            print(f"Enhanced Web Search Results: {len(search_results)}")
            
            for i, result in enumerate(search_results[:3], 1):
                print(f"  {i}. {result.title[:60]}...")
                print(f"     URL: {result.url}")
                print(f"     Relevance: {result.relevance_score:.3f}")
                
        except Exception as e:
            print(f"❌ Enhanced Search Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Step 4: Test full perplexity search
        print(f"\n🎯 Testing full perplexity_search...")
        try:
            response = await service.perplexity_search(
                query=query,
                max_results=8,
                synthesize_answer=False,
                include_recent=True
            )
            print(f"Full Search Results: {len(response.sources)}")
            print(f"Search Time: {response.search_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Full Search Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_search_flow())