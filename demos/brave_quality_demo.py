#!/usr/bin/env python3
"""
Demo: Brave Search as High-Quality Source Integration
Shows how the system prioritizes and cites Brave sources as high-quality.
"""

import asyncio
import os
from app.services.perplexity_web_search import perplexity_web_search

def display_results(result, title):
    """Display search results with quality indicators."""
    print(f"\n🔍 {title}")
    print("=" * 60)
    print(f"Query: {result['query']}")
    print(f"Sources found: {len(result['sources'])}")
    print(f"Confidence: {result.get('confidence_score', 0):.2f}")
    print(f"Search time: {result.get('search_time', 0):.2f}s")
    
    print("\n📚 High-Quality Citations:")
    citations = result.get('citations', {})
    for cid, citation in citations.items():
        quality_indicator = "🟢 HIGH QUALITY" if "[High-Quality Source]" in citation else "🔍 Standard"
        print(f"  [{cid}] {quality_indicator}")
        print(f"      {citation}")
    
    print("\n📋 Source Breakdown:")
    brave_count = sum(1 for s in result['sources'] if s.get('source') == 'brave_search')
    ddgs_count = len(result['sources']) - brave_count
    
    if brave_count > 0:
        print(f"  🟠 Brave Search (High-Quality): {brave_count} sources")
    if ddgs_count > 0:
        print(f"  🔍 DuckDuckGo (Standard): {ddgs_count} sources")
    
    print(f"\n💬 Answer Preview:")
    answer = result.get('answer', '')
    preview = answer[:300] + "..." if len(answer) > 300 else answer
    print(f"   {preview}")

def main():
    """Demonstrate Brave search as high-quality source."""
    print("🚀 Brave Search High-Quality Source Integration Demo")
    print("=" * 70)
    
    # Check if Brave API is configured
    brave_key = os.getenv("BRAVE_API_KEY")
    if brave_key:
        print("✅ Brave Search API configured - High-quality sources enabled")
    else:
        print("⚠️  Brave Search API not configured - Using fallback sources only")
        print("   To enable high-quality Brave sources:")
        print("   1. Get API key: https://api.search.brave.com/app/keys")
        print("   2. Add to .env: BRAVE_API_KEY=your_key_here")
    
    # Test queries that benefit from high-quality sources
    test_queries = [
        "Apple stock price forecast 2025",
        "Microsoft earnings report Q3 2025",
        "Tesla market analysis recent developments"
    ]
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n{'='*20} Test {i} {'='*20}")
            result = perplexity_web_search(
                query=query,
                max_results=3,
                synthesize_answer=True,
                include_recent=True
            )
            
            display_results(result, f"Test {i}: {query}")
            
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
    
    print("\n" + "="*70)
    print("🎯 Key Benefits of Brave as High-Quality Source:")
    print("   • Higher relevance scores (typically 0.8-1.0)")
    print("   • Better content quality and accuracy")  
    print("   • Time-based filtering for recent information")
    print("   • Clear source attribution in citations")
    print("   • Visual quality indicators in frontend")
    print("   • Prioritized in search result ordering")
    
    if not brave_key:
        print("\n💡 Enable Brave Search for enhanced quality:")
        print("   Free tier: 2,000 queries/month")
        print("   Perfect for high-quality financial and market research")

if __name__ == "__main__":
    main()