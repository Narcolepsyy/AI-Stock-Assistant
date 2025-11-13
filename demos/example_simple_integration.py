"""
Example integration of simplified Brave Search into main application
"""
import asyncio
from app.services.simple_brave_search import simple_brave_search

# Example: Using simplified Brave Search in your application
async def get_financial_news(query: str, country: str = None) -> list:
    """
    Get financial news using simplified Brave Search
    
    Args:
        query: Search query (e.g., "Tesla earnings", "Apple stock")
        country: Country code for localized results (e.g., "US", "JP", "GB")
    
    Returns:
        List of search results
    """
    try:
        results = await simple_brave_search(
            query=query,
            count=10,
            country=country
        )
        
        # You can add additional filtering here if needed
        financial_results = []
        for result in results:
            # Basic financial site filtering (optional)
            financial_domains = [
                'yahoo.com', 'cnbc.com', 'bloomberg.com', 'reuters.com',
                'marketwatch.com', 'morningstar.com', 'fool.com',
                'finance.yahoo.com', 'investors.com'
            ]
            
            url = result.get('url', '').lower()
            is_financial = any(domain in url for domain in financial_domains)
            
            # Include all results, but mark financial ones
            result['is_financial_source'] = is_financial
            financial_results.append(result)
        
        return financial_results
    
    except Exception as e:
        print(f"Error getting financial news: {e}")
        return []

# Example usage
async def demo_usage():
    """Demonstrate how to use the simplified Brave Search"""
    
    print("📊 Financial News Search Examples")
    print("=" * 50)
    
    # Example 1: US-specific Tesla news
    print("\n1. Tesla news (US):")
    us_tesla = await get_financial_news("Tesla quarterly earnings", "US")
    for result in us_tesla[:3]:
        print(f"   • {result['title']}")
        print(f"     {result['url']}")
        if result['is_financial_source']:
            print("     📈 Financial source")
        print()
    
    # Example 2: Global Apple news
    print("\n2. Apple news (Global):")
    global_apple = await get_financial_news("Apple financial results")
    for result in global_apple[:3]:
        print(f"   • {result['title']}")
        print(f"     {result['url']}")
        if result['is_financial_source']:
            print("     📈 Financial source")
        print()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_usage())