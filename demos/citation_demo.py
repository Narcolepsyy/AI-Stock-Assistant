#!/usr/bin/env python3
"""
Citation System Demo - Shows how web sources are properly cited
"""
import asyncio
import sys
import re
from typing import Dict, Any
from app.services.perplexity_web_search import PerplexityWebSearchService

def format_citations_for_display(answer: str, citations: Dict[int, str]) -> str:
    """Format answer with proper citation display."""
    
    # Add citations section at the end
    formatted_answer = answer + "\n\n"
    
    if citations:
        formatted_answer += "## 📚 Sources\n\n"
        for cite_id, cite_info in sorted(citations.items()):
            formatted_answer += f"**[{cite_id}]** {cite_info}\n\n"
    
    return formatted_answer

def extract_citation_stats(answer: str, citations: Dict[int, str]) -> Dict[str, Any]:
    """Extract statistics about citation usage."""
    
    # Find all citations in the text
    citations_in_text = re.findall(r'\[(\d+)\]', answer)
    unique_citations_used = set(citations_in_text)
    
    # Count citation frequency
    citation_frequency = {}
    for cite in citations_in_text:
        citation_frequency[int(cite)] = citation_frequency.get(int(cite), 0) + 1
    
    return {
        'total_citations_in_text': len(citations_in_text),
        'unique_citations_used': len(unique_citations_used),
        'available_citations': len(citations),
        'citation_coverage': len(unique_citations_used) / len(citations) if citations else 0,
        'citation_frequency': citation_frequency,
        'citation_ids_used': sorted(list(unique_citations_used))
    }

async def demo_japanese_financial_search():
    """Demo search with Japanese financial query to show citation handling."""
    
    print("🇯🇵 Testing Japanese Financial Query with Citations")
    print("="*60)
    
    service = PerplexityWebSearchService()
    
    # Japanese financial query
    query = "日本銀行の金利政策 最新"
    result = await service.perplexity_search(query, max_results=6)
    
    print(f"📝 Query: {result.query}")
    print(f"⏱️  Search Time: {result.search_time:.2f}s")
    print(f"⏱️  Synthesis Time: {result.synthesis_time:.2f}s")
    print(f"🎯 Confidence: {result.confidence_score:.2f}")
    print(f"📊 Sources: {len(result.sources)}")
    
    # Analyze citations
    stats = extract_citation_stats(result.answer, result.citations)
    
    print(f"\n📈 Citation Statistics:")
    print(f"   • Total citations used: {stats['total_citations_in_text']}")
    print(f"   • Unique sources cited: {stats['unique_citations_used']}")
    print(f"   • Available sources: {stats['available_citations']}")
    print(f"   • Citation coverage: {stats['citation_coverage']:.1%}")
    
    print(f"\n📖 Answer with Citations:")
    print("-" * 60)
    formatted = format_citations_for_display(result.answer, result.citations)
    print(formatted)
    
    return result, stats

async def demo_english_tech_search():
    """Demo search with English tech query."""
    
    print("\n🔍 Testing English Tech Query with Citations")
    print("="*60)
    
    service = PerplexityWebSearchService()
    
    # English tech query
    query = "OpenAI GPT-4 latest features and improvements 2024"
    result = await service.perplexity_search(query, max_results=6)
    
    print(f"📝 Query: {result.query}")
    print(f"⏱️  Total Time: {result.total_time:.2f}s")
    print(f"🎯 Confidence: {result.confidence_score:.2f}")
    
    # Analyze citations
    stats = extract_citation_stats(result.answer, result.citations)
    
    print(f"\n📈 Citation Statistics:")
    print(f"   • Citations in text: {stats['total_citations_in_text']}")
    print(f"   • Unique sources: {stats['unique_citations_used']}")
    print(f"   • Coverage: {stats['citation_coverage']:.1%}")
    
    print(f"\n🔗 Source Mapping:")
    for cite_id, frequency in stats['citation_frequency'].items():
        source_info = result.citations.get(cite_id, 'Unknown')
        print(f"   [{cite_id}] Used {frequency}x - {source_info}")
    
    return result, stats

async def main():
    """Run citation system demonstrations."""
    
    print("🌐 Web Search Citation System Demo")
    print("="*60)
    
    try:
        # Demo 1: Japanese financial query
        japanese_result, japanese_stats = await demo_japanese_financial_search()
        
        # Demo 2: English tech query
        english_result, english_stats = await demo_english_tech_search()
        
        print(f"\n📊 Summary Comparison:")
        print(f"Japanese Query:")
        print(f"   • Sources found: {len(japanese_result.sources)}")
        print(f"   • Citations used: {japanese_stats['total_citations_in_text']}")
        print(f"   • Answer length: {len(japanese_result.answer)} chars")
        
        print(f"English Query:")
        print(f"   • Sources found: {len(english_result.sources)}")
        print(f"   • Citations used: {english_stats['total_citations_in_text']}")
        print(f"   • Answer length: {len(english_result.answer)} chars")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())