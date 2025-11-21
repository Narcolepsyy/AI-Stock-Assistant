#!/usr/bin/env python3
"""
Test script for stock prediction service.
Tests model training and prediction without GPU (CPU fallback).
"""

import sys
import os
from pathlib import Path

# Disable GPU for testing to avoid CUDA errors in CI/CD
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.stock_prediction_service import (
    train_model,
    predict_stock_price,
    get_model_info,
    list_available_models,
    evaluate_model_accuracy,
)


def test_training():
    """Test model training with minimal data."""
    print("=" * 60)
    print("TEST 1: Model Training")
    print("=" * 60)
    
    try:
        # Use small dataset for quick testing
        result = train_model(
            symbol="AAPL",
            period="1y",  # 1 year for faster training
            save_model=True
        )
        
        print(f"✓ Training completed!")
        print(f"  Train Loss: {result['metrics']['train_loss']:.6f}")
        print(f"  Val Loss:   {result['metrics']['val_loss']:.6f}")
        print(f"  Train MAE:  {result['metrics']['train_mae']:.6f}")
        print(f"  Val MAE:    {result['metrics']['val_mae']:.6f}")
        print(f"  Model Path: {result['model_path']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return False


def test_prediction():
    """Test price prediction."""
    print("\n" + "=" * 60)
    print("TEST 2: Price Prediction")
    print("=" * 60)
    
    try:
        result = predict_stock_price(
            symbol="AAPL",
            days=7,
            auto_train=False  # Use existing model from test 1
        )
        
        print(f"✓ Prediction completed!")
        print(f"  Symbol: {result['symbol']}")
        print(f"  Current Price: ${result['current_price']}")
        print(f"  Predicted (7d): ${result['summary']['final_predicted_price']}")
        print(f"  Change: {result['summary']['total_change']:+.2f} ({result['summary']['total_change_pct']:+.2f}%)")
        print(f"  Trend: {result['summary']['trend_en']}")
        
        print(f"\n  Predictions:")
        for pred in result['predictions'][:3]:  # Show first 3 days
            print(f"    {pred['date']}: ${pred['predicted_price']} ({pred['change_pct']:+.2f}%)")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_info():
    """Test model info retrieval."""
    print("\n" + "=" * 60)
    print("TEST 3: Model Info")
    print("=" * 60)
    
    try:
        result = get_model_info("AAPL")
        
        if result['model_exists']:
            print(f"✓ Model found!")
            config = result['config']
            print(f"  Symbol: {config['symbol']}")
            print(f"  Trained: {config['trained_date'][:10]}")
            print(f"  Val MAE: {config['val_mae']:.4f}")
            print(f"  Features: {', '.join(config['features'])}")
        else:
            print(f"  No model found for AAPL")
        
        return True
        
    except Exception as e:
        print(f"✗ Model info failed: {e}")
        return False


def test_list_models():
    """Test listing all models."""
    print("\n" + "=" * 60)
    print("TEST 4: List Models")
    print("=" * 60)
    
    try:
        result = list_available_models()
        
        print(f"✓ Found {result['count']} model(s)")
        
        for model in result['models']:
            print(f"  • {model['symbol']:10s} - {model['trained_date'][:10]} - MAE: {model['val_mae']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ List models failed: {e}")
        return False


def test_accuracy_check():
    """Verify accuracy metrics on recent historical data."""
    print("\n" + "=" * 60)
    print("TEST 5: Accuracy Verification")
    print("=" * 60)

    try:
        result = evaluate_model_accuracy("AAPL", period="6mo")
        metrics = result['metrics']

        print("✓ Accuracy evaluation completed!")
        print(f"  Samples: {result['samples']}")
        print(f"  MAE (returns):  {metrics['mae_return']*100:.2f}%")
        print(f"  RMSE (returns): {metrics['rmse_return']*100:.2f}%")
        print(f"  MAPE: {metrics['mape_pct']:.2f}%")

        if result['recent_examples']:
            print("\n  Recent examples:")
            for item in result['recent_examples']:
                print(
                    f"    Actual Return: {item['actual_return_pct']:+.2f}% | "
                    f"Predicted: {item['predicted_return_pct']:+.2f}% | "
                    f"Error: {item['error_pct']:+.2f}%"
                )

        return True

    except Exception as e:
        print(f"✗ Accuracy evaluation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n🧪 Stock Prediction Service Tests")
    print("Running on CPU (GPU disabled for testing)\n")
    
    results = []
    
    # Test 1: Training
    results.append(("Training", test_training()))
    
    # Test 2: Prediction (requires model from test 1)
    results.append(("Prediction", test_prediction()))
    
    # Test 3: Model Info
    results.append(("Model Info", test_model_info()))
    
    # Test 4: List Models
    results.append(("List Models", test_list_models()))

    # Test 5: Accuracy Verification
    results.append(("Accuracy Verification", test_accuracy_check()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
