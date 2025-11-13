#!/usr/bin/env python3
"""Debug script to test the web search functionality step by step."""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.perplexity_web_search import perplexity_web_search

def test_direct_search():
    """Test direct perplexity search functionality."""
    print("🔍 Testing direct perplexity search...")
    
    query = "Hachijuni Bank merger with Nagano Bank"
    print(f"Query: {query}")
    
    try:
        result = perplexity_web_search(
            query=query,
            max_results=5,
            synthesize_answer=False,
            include_recent=True
        )
        
        print(f"Result type: {type(result)}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict):
            print(f"Query processed: {result.get('query', 'N/A')}")
            print(f"Method: {result.get('method', 'N/A')}")
            print(f"Sources count: {len(result.get('sources', []))}")
            
            sources = result.get('sources', [])
            for i, source in enumerate(sources[:3], 1):
                print(f"\n{i}. {source.get('title', 'No title')}")
                print(f"   URL: {source.get('url', 'No URL')}")
                print(f"   Snippet: {source.get('snippet', 'No snippet')[:100]}...")
        else:
            print(f"Unexpected result format: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_japanese_query():
    """Test Japanese query processing."""
    print("\n🔍 Testing Japanese query...")
    
    query = "長野銀行との統合関連ニュース"
    print(f"Query: {query}")
    
    try:
        result = perplexity_web_search(
            query=query,
            max_results=5,
            synthesize_answer=False,
            include_recent=True
        )
        
        print(f"Query processed: {result.get('query', 'N/A')}")
        print(f"Sources count: {len(result.get('sources', []))}")
        
        sources = result.get('sources', [])
        for i, source in enumerate(sources[:3], 1):
            print(f"\n{i}. {source.get('title', 'No title')}")
            print(f"   URL: {source.get('url', 'No URL')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_search()
    test_japanese_query()