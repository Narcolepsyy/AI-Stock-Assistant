#!/usr/bin/env python3
"""Simple validation test for enhanced Perplexity search."""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_basic_functionality():
    """Test basic functionality of enhanced search."""
    print("🔍 Testing Enhanced Perplexity Search...")
    
    try:
        from app.services.perplexity_web_search import perplexity_web_search
        
        # Simple test query
        query = "Tesla stock price"
        print(f"Testing query: {query}")
        
        result = perplexity_web_search(
            query=query,
            max_results=3,
            synthesize_answer=True,
            include_recent=True
        )
        
        print(f"✅ Search completed successfully!")
        print(f"Query: {result.get('query')}")
        print(f"Sources found: {len(result.get('sources', []))}")
        print(f"Answer generated: {'Yes' if result.get('answer') else 'No'}")
        print(f"Confidence: {result.get('confidence_score', 0):.3f}")
        print(f"Total time: {result.get('total_time', 0):.2f}s")
        
        # Show enhanced scoring
        sources = result.get('sources', [])
        if sources:
            print(f"\nEnhanced Scoring Results:")
            print(f"{'#':<3} {'BM25':<8} {'Semantic':<8} {'Combined':<8} {'Title'}")
            print("-" * 60)
            
            for i, source in enumerate(sources[:3]):
                print(f"{i+1:<3} {source.get('bm25_score', 0):.3f}    {source.get('semantic_score', 0):.3f}   {source.get('combined_score', 0):.3f}   {source.get('title', '')[:30]}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_dependencies():
    """Test if all dependencies are available."""
    print("\n🔧 Testing Dependencies...")
    
    # Test BM25
    try:
        from rank_bm25 import BM25Okapi
        print("✅ rank-bm25 available")
    except ImportError:
        print("❌ rank-bm25 not available")
        return False
    
    # Test optional dependencies
    try:
        import numpy as np
        print("✅ numpy available")
    except ImportError:
        print("⚠️  numpy not available (using fallback)")
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ scikit-learn available")
    except ImportError:
        print("⚠️  scikit-learn not available (using fallback)")
    
    # Test Azure configuration
    try:
        from app.core.config import (
            AZURE_OPENAI_DEPLOYMENT_OSS_120B,
            AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
            AZURE_OPENAI_API_KEY
        )
        
        if AZURE_OPENAI_DEPLOYMENT_OSS_120B:
            print(f"✅ Azure GPT OSS 120B configured: {AZURE_OPENAI_DEPLOYMENT_OSS_120B}")
        else:
            print("⚠️  Azure GPT OSS 120B not configured")
        
        if AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT:
            print(f"✅ Azure Embeddings configured: {AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT}")
        else:
            print("⚠️  Azure Embeddings not configured")
            
        if AZURE_OPENAI_API_KEY:
            print("✅ Azure API key configured")
        else:
            print("❌ Azure API key not configured")
            return False
            
    except ImportError as e:
        print(f"❌ Configuration import error: {e}")
        return False
    
    return True

async def main():
    """Run validation tests."""
    print("Enhanced Perplexity Search - Validation Test")
    print("=" * 50)
    
    # Test dependencies first
    deps_ok = await test_dependencies()
    if not deps_ok:
        print("\n❌ Dependencies check failed!")
        return False
    
    # Test basic functionality
    basic_ok = await test_basic_functionality()
    if not basic_ok:
        print("\n❌ Basic functionality test failed!")
        return False
    
    print("\n🎉 Validation completed successfully!")
    print("\nEnhancements added:")
    print("  • BM25 lexical ranking")
    print("  • Azure text embeddings semantic similarity")
    print("  • Hybrid scoring system")
    print("  • Azure GPT OSS 120B for answer synthesis")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)