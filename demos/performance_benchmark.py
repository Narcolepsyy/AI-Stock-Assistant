"""
Performance test suite to measure and validate optimizations.
Tests web search performance, memory usage, and response times.
"""

import asyncio
import time
import sys
import os
import psutil
import gc
from datetime import datetime
from typing import Dict, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.enhanced_rag_service import EnhancedRAGService
from app.services.perplexity_web_search import perplexity_web_search


class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_time = time.time()
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        elapsed_time = time.time() - self.start_time
        
        return {
            'memory_mb': current_memory,
            'memory_increase_mb': current_memory - self.start_memory,
            'elapsed_seconds': elapsed_time,
            'cpu_percent': self.process.cpu_percent()
        }


async def test_search_method_performance():
    """Test performance of different search methods."""
    
    print("🚀 Performance Testing Suite")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "Apple stock price AAPL",
        "Tesla earnings report 2024", 
        "Microsoft financial performance",
        "Amazon AWS revenue growth",
        "Google Alphabet stock analysis"
    ]
    
    search_methods = [
        ("Optimized Alternative Search", test_optimized_search),
        ("Enhanced RAG Service", test_enhanced_rag),
        ("Robust Web Search Tool", test_robust_search),
        ("Original Alternative Search", test_original_search),
    ]
    
    results = {}
    
    for method_name, method_func in search_methods:
        print(f"\n📊 Testing: {method_name}")
        print("-" * 40)
        
        method_results = []
        monitor = PerformanceMonitor()
        
        for query in test_queries:
            print(f"  🔍 Query: '{query[:30]}...'", end=" ")
            
            start_time = time.time()
            try:
                result = await method_func(query)
                end_time = time.time()
                
                duration = end_time - start_time
                count = result.get('count', 0) if isinstance(result, dict) else len(result.get('combined_chunks', []))
                
                method_results.append({
                    'query': query,
                    'duration': duration,
                    'result_count': count,
                    'success': True
                })
                
                print(f"✅ {duration:.2f}s ({count} results)")
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                method_results.append({
                    'query': query,
                    'duration': duration,
                    'result_count': 0,
                    'success': False,
                    'error': str(e)
                })
                
                print(f"❌ {duration:.2f}s (ERROR: {str(e)[:30]}...)")
        
        # Calculate summary statistics
        successful_queries = [r for r in method_results if r['success']]
        if successful_queries:
            avg_duration = sum(r['duration'] for r in successful_queries) / len(successful_queries)
            avg_results = sum(r['result_count'] for r in successful_queries) / len(successful_queries)
            success_rate = len(successful_queries) / len(test_queries) * 100
        else:
            avg_duration = float('inf')
            avg_results = 0
            success_rate = 0
        
        final_metrics = monitor.get_current_metrics()
        
        results[method_name] = {
            'avg_duration': avg_duration,
            'avg_results': avg_results,
            'success_rate': success_rate,
            'memory_increase': final_metrics['memory_increase_mb'],
            'total_memory': final_metrics['memory_mb'],
            'details': method_results
        }
        
        print(f"  📈 Avg Duration: {avg_duration:.2f}s")
        print(f"  📊 Avg Results: {avg_results:.1f}")
        print(f"  ✅ Success Rate: {success_rate:.1f}%")
        print(f"  🧠 Memory: +{final_metrics['memory_increase_mb']:.1f}MB")
        
        # Force garbage collection between tests
        gc.collect()
        await asyncio.sleep(1)  # Brief pause between tests
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    
    # Sort methods by performance (fastest avg duration first)
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['avg_duration'])
    
    for i, (method_name, stats) in enumerate(sorted_methods, 1):
        icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
        print(f"{icon} {i}. {method_name}")
        print(f"    ⏱️  Avg Duration: {stats['avg_duration']:.2f}s")
        print(f"    📊 Avg Results: {stats['avg_results']:.1f}")
        print(f"    ✅ Success Rate: {stats['success_rate']:.1f}%")
        print(f"    🧠 Memory Impact: +{stats['memory_increase']:.1f}MB")
        
        # Performance grade
        if stats['avg_duration'] < 1.0 and stats['success_rate'] > 90:
            grade = "A+ (Excellent)"
        elif stats['avg_duration'] < 2.0 and stats['success_rate'] > 80:
            grade = "A (Good)"
        elif stats['avg_duration'] < 3.0 and stats['success_rate'] > 70:
            grade = "B (Average)"
        else:
            grade = "C (Needs Improvement)"
        
        print(f"    🎯 Grade: {grade}")
        print()
    
    return results


async def test_optimized_search(query: str):
    """Test optimized alternative search."""
    return await optimized_alternative_search(query, max_results=5)


async def test_enhanced_rag(query: str):
    """Test enhanced RAG service."""
    service = EnhancedRAGService()
    return await service.augmented_search(query, kb_k=2, web_results=5, max_total_chunks=5)


async def test_robust_search(query: str):
    """Test robust web search tool (synchronous)."""
    # This is synchronous, so we need to run it in a thread
    import concurrent.futures
    
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor, 
            _robust_web_search, 
            query, 5, False  # max_results=5, include_content=False
        )
    return result


async def test_original_search(query: str):
    """Test original alternative search."""
    # This uses the synchronous wrapper, so run in thread
    import concurrent.futures
    
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            alternative_web_search,
            query, 5
        )
    return result


async def test_memory_leak_detection():
    """Test for memory leaks in repeated searches."""
    print("\n🧠 Memory Leak Detection Test")
    print("-" * 40)
    
    monitor = PerformanceMonitor()
    query = "Apple stock price test"
    
    # Baseline memory
    initial_metrics = monitor.get_current_metrics()
    print(f"Initial memory: {initial_metrics['memory_mb']:.1f}MB")
    
    # Run 20 searches to see if memory increases
    for i in range(20):
        try:
            await optimized_alternative_search(query, max_results=3)
            if i % 5 == 4:  # Every 5th iteration
                current_metrics = monitor.get_current_metrics()
                print(f"After {i+1} searches: {current_metrics['memory_mb']:.1f}MB (+{current_metrics['memory_increase_mb']:.1f}MB)")
        except Exception as e:
            print(f"Search {i+1} failed: {e}")
    
    # Force garbage collection
    gc.collect()
    await asyncio.sleep(1)
    
    # Final memory check
    final_metrics = monitor.get_current_metrics()
    print(f"Final memory: {final_metrics['memory_mb']:.1f}MB (+{final_metrics['memory_increase_mb']:.1f}MB total)")
    
    # Memory leak assessment
    if final_metrics['memory_increase_mb'] < 10:
        print("✅ No significant memory leak detected")
    elif final_metrics['memory_increase_mb'] < 25:
        print("⚠️  Minor memory increase detected")
    else:
        print("❌ Potential memory leak detected!")
    
    return final_metrics


async def run_all_tests():
    """Run complete performance test suite."""
    start_time = time.time()
    
    try:
        # Run main performance tests
        performance_results = await test_search_method_performance()
        
        # Run memory leak test
        memory_results = await test_memory_leak_detection()
        
        # Cleanup any remaining sessions
        try:
            await OptimizedAlternativeWebSearch.cleanup_session()
        except:
            pass
        
        total_time = time.time() - start_time
        
        print(f"\n🎯 Testing completed in {total_time:.1f}s")
        print("=" * 60)
        
        # Generate recommendations
        print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
        
        best_method = min(performance_results.items(), key=lambda x: x[1]['avg_duration'])
        print(f"• Use '{best_method[0]}' for best performance")
        
        if memory_results['memory_increase_mb'] > 10:
            print("• Consider implementing session pooling for memory optimization")
        
        fast_methods = [name for name, stats in performance_results.items() 
                       if stats['avg_duration'] < 2.0 and stats['success_rate'] > 80]
        if fast_methods:
            print(f"• Recommended methods: {', '.join(fast_methods)}")
        
        print("\n✨ Performance testing complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())