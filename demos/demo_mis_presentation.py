#!/usr/bin/env python3
"""
MIS (Management Information System) Demo Script
Demonstrates AI-powered financial analysis capabilities for business intelligence
"""

import asyncio
import time
from app.services.perplexity_web_search import PerplexityWebSearchService

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{title}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

def print_query(query: str, query_num: int, total: int):
    """Print formatted query"""
    print(f"{Colors.OKCYAN}[Query {query_num}/{total}]{Colors.ENDC} {Colors.BOLD}{query}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'-'*80}{Colors.ENDC}")

def print_result(result: dict, elapsed_time: float):
    """Print formatted search results"""
    if result.get('error'):
        print(f"{Colors.FAIL}❌ Error: {result['error']}{Colors.ENDC}\n")
        return
    
    # Status indicator
    if elapsed_time < 3.0:
        status = f"{Colors.OKGREEN}✅ EXCELLENT{Colors.ENDC}"
    elif elapsed_time < 5.0:
        status = f"{Colors.WARNING}⚡ GOOD{Colors.ENDC}"
    else:
        status = f"{Colors.WARNING}⏱️  ACCEPTABLE{Colors.ENDC}"
    
    print(f"🚀 Status: {status}")
    print(f"⏱️  Time: {elapsed_time:.2f}s")
    
    # Answer preview
    answer = result.get('answer', '')
    if answer:
        preview = answer[:200] + "..." if len(answer) > 200 else answer
        print(f"\n📝 Answer Preview:")
        print(f"{Colors.OKGREEN}{preview}{Colors.ENDC}")
    
    # Sources
    sources = result.get('sources', [])
    if sources:
        print(f"\n📄 Sources: {len(sources)} high-quality results")
        for i, source in enumerate(sources[:3], 1):
            title = source.get('title', 'No title')[:70]
            url = source.get('url', 'No URL')
            score = source.get('combined_score', source.get('relevance_score', 0))
            print(f"  [{i}] {title}...")
            print(f"      URL: {url}")
            print(f"      Score: {score:.3f}")
    
    print()

async def run_demo():
    """Run MIS presentation demo with various business intelligence queries"""
    
    print_section("🎓 MIS DEMONSTRATION - AI-Powered Financial Analysis System")
    print(f"{Colors.OKGREEN}Demonstrating: Web Search, Data Analysis, and Business Intelligence{Colors.ENDC}\n")
    
    # Initialize service
    service = PerplexityWebSearchService()
    
    # Demo queries covering different MIS use cases
    demo_queries = [
        {
            "category": "📊 Market Intelligence",
            "queries": [
                "What are the latest trends in artificial intelligence investments 2025?",
                "Tesla stock analyst predictions and price targets",
            ]
        },
        {
            "category": "🏢 Corporate Analysis",
            "queries": [
                "Microsoft earnings report Q4 2024 analysis",
                "Apple's latest product launches and market impact",
            ]
        },
        {
            "category": "💹 Financial Technology",
            "queries": [
                "Latest developments in cryptocurrency regulation",
                "Fintech industry growth forecast 2025",
            ]
        },
        {
            "category": "🌏 Japanese Market (日本市場)",
            "queries": [
                "住信SBIネット銀行の最新情報",
                "日本の半導体産業の最新動向",
            ]
        }
    ]
    
    total_queries = sum(len(cat['queries']) for cat in demo_queries)
    query_count = 0
    total_time = 0
    results_summary = []
    
    # Run all demo queries
    for category_data in demo_queries:
        category = category_data['category']
        queries = category_data['queries']
        
        print_section(category)
        
        for query in queries:
            query_count += 1
            print_query(query, query_count, total_queries)
            
            start_time = time.time()
            try:
                # Execute search with fast mode
                result = await service.perplexity_search(
                    query=query,
                    max_results=5,
                    include_recent=True
                )
                
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                # Convert to dict for display
                result_dict = {
                    'answer': result.answer if result else '',
                    'sources': [
                        {
                            'title': s.title,
                            'url': s.url,
                            'combined_score': s.combined_score,
                            'relevance_score': s.relevance_score
                        }
                        for s in (result.sources if result else [])
                    ],
                    'citations': result.citations if result else {}
                }
                
                print_result(result_dict, elapsed_time)
                
                results_summary.append({
                    'query': query,
                    'time': elapsed_time,
                    'sources': len(result_dict['sources']),
                    'success': True
                })
                
            except Exception as e:
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                print_result({'error': str(e)}, elapsed_time)
                
                results_summary.append({
                    'query': query,
                    'time': elapsed_time,
                    'sources': 0,
                    'success': False
                })
            
            # Brief pause between queries
            await asyncio.sleep(0.5)
    
    # Final summary
    print_section("📊 PERFORMANCE SUMMARY")
    
    successful = sum(1 for r in results_summary if r['success'])
    avg_time = total_time / len(results_summary) if results_summary else 0
    
    for i, result in enumerate(results_summary, 1):
        status_icon = "✅" if result['success'] else "❌"
        status_text = f"{result['time']:.2f}s"
        query_preview = result['query'][:60] + "..." if len(result['query']) > 60 else result['query']
        print(f"{status_icon} {status_text:>6} | {result['sources']:2} sources | {query_preview}")
    
    print(f"\n{Colors.HEADER}{'-'*80}{Colors.ENDC}")
    print(f"Total Queries: {len(results_summary)}")
    print(f"Successful: {successful}/{len(results_summary)}")
    print(f"Average Time: {avg_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    
    # Performance rating
    if avg_time < 3.0:
        rating = f"{Colors.OKGREEN}✅ EXCELLENT! System meets ChatGPT/Perplexity speed standards.{Colors.ENDC}"
    elif avg_time < 5.0:
        rating = f"{Colors.WARNING}⚡ GOOD! Close to target performance.{Colors.ENDC}"
    else:
        rating = f"{Colors.WARNING}⏱️  ACCEPTABLE but can be optimized further.{Colors.ENDC}"
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(rating)
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    # Key features demonstrated
    print_section("🎯 KEY MIS CAPABILITIES DEMONSTRATED")
    print(f"""
{Colors.OKGREEN}✅ Real-time Information Retrieval{Colors.ENDC}
   - Fast web search with <3s response time target
   - Multi-source aggregation (Brave Search + DuckDuckGo)

{Colors.OKGREEN}✅ Business Intelligence{Colors.ENDC}
   - Market trends analysis
   - Corporate financial analysis
   - Industry sector insights

{Colors.OKGREEN}✅ Multi-language Support{Colors.ENDC}
   - English and Japanese query processing
   - Localized financial market data

{Colors.OKGREEN}✅ Source Attribution{Colors.ENDC}
   - Citation tracking for all information
   - Quality scoring for sources
   - Transparency in data sources

{Colors.OKGREEN}✅ High-Performance Architecture{Colors.ENDC}
   - Parallel search execution
   - Smart caching strategies
   - Optimized semantic ranking
    """)

if __name__ == "__main__":
    print(f"\n{Colors.BOLD}Starting MIS Demo Presentation...{Colors.ENDC}\n")
    asyncio.run(run_demo())
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}Demo Complete!{Colors.ENDC}\n")
