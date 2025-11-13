#!/usr/bin/env python3
"""
Test script for free news service.
Tests RSS feed fetching and news aggregation without API keys.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import free_news_service
import json


def test_rss_feeds():
    """Test fetching from RSS feeds."""
    print("=" * 60)
    print("Testing RSS Feed Fetching")
    print("=" * 60)
    
    categories = ["financial", "crypto", "commodities"]
    
    for category in categories:
        print(f"\n📰 Fetching {category} news from RSS feeds...")
        articles = free_news_service.fetch_rss_feeds(category, limit=5)
        
        if articles:
            print(f"✅ Found {len(articles)} articles")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n{i}. {article['title'][:80]}...")
                print(f"   Source: {article['source']}")
                print(f"   URL: {article['url'][:60]}...")
        else:
            print(f"❌ No articles found for {category}")


def test_market_news():
    """Test the main market news aggregation function."""
    print("\n\n" + "=" * 60)
    print("Testing Market News Aggregation")
    print("=" * 60)
    
    print("\n📊 Fetching all market news...")
    news_by_category = free_news_service.get_market_news(
        categories=["financial", "crypto", "commodities"],
        limit_per_category=5
    )
    
    total_articles = 0
    for category, articles in news_by_category.items():
        print(f"\n📂 {category.upper()}")
        print(f"   Found {len(articles)} articles")
        total_articles += len(articles)
        
        if articles:
            print(f"   Latest: {articles[0]['title'][:70]}...")
    
    print(f"\n✅ Total articles across all categories: {total_articles}")


def test_summary_generation():
    """Test summary generation from news articles."""
    print("\n\n" + "=" * 60)
    print("Testing Summary Generation")
    print("=" * 60)
    
    # Fetch some articles
    print("\n📰 Fetching financial news...")
    articles = free_news_service.fetch_rss_feeds("financial", limit=5)
    
    if not articles:
        print("❌ No articles available for testing")
        return
    
    print(f"✅ Fetched {len(articles)} articles")
    
    # Generate summary
    print("\n📝 Generating summary...")
    summary = free_news_service.generate_summary_from_news(
        articles,
        "Test Financial Summary"
    )
    
    if summary:
        print(f"✅ Summary generated successfully")
        print(f"\n{json.dumps(summary, indent=2)}")
    else:
        print("❌ Failed to generate summary")


def main():
    print("\n🚀 Free News Service Test Suite")
    print("=" * 60)
    print("This test uses only FREE RSS feeds - no API keys required!")
    print("=" * 60)
    
    try:
        # Test 1: RSS Feeds
        test_rss_feeds()
        
        # Test 2: Market News Aggregation
        test_market_news()
        
        # Test 3: Summary Generation
        test_summary_generation()
        
        print("\n\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\n📌 Note: If some feeds failed, it's normal - RSS feeds")
        print("   can be temporarily unavailable. The system will use")
        print("   whatever feeds are accessible.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
