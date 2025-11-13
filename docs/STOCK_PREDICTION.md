# Stock Price Prediction - AI/ML Integration

## Overview

Lightweight LSTM-based stock price prediction system optimized for **GTX 1650 Ti Mobile (4GB VRAM)**. Integrates seamlessly with the AI Stocks Assistant chat interface and provides REST API endpoints.

## Features

✅ **Lightweight Architecture**
- 3-layer LSTM with 64→32→16 units (model size < 50MB)
- GPU memory growth enabled (no OOM on 4GB VRAM)
- Inference time: < 1 second on GPU
- CPU fallback supported

✅ **Technical Features**
- Uses 7 technical indicators: close, volume, high, low, SMA(5), SMA(20), RSI(14)
- 60-day lookback window for predictions
- Predicts 1-30 days into the future
- Auto-adjusts for weekends (stock market closed)

✅ **Model Management**
- Auto-training when model not found
- Per-symbol model storage
- Training metrics tracking (loss, MAE)
- Model configuration persistence

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow>=2.15.0,<2.16.0 scikit-learn>=1.3.0
```

**Note:** TensorFlow 2.15.x is optimized for GTX 1650 Ti. Uses ~2-3GB VRAM during training.

### 2. Train Your First Model

#### Via Command Line Script
```bash
# Train single stock
python scripts/train_prediction_model.py AAPL

# Train with 5 years of data
python scripts/train_prediction_model.py AAPL --period 5y

# Train all popular stocks
python scripts/train_prediction_model.py --all

# List trained models
python scripts/train_prediction_model.py --list
```

#### Via REST API
```bash
# Start server
uvicorn main:app --reload

# Train model (requires authentication)
curl -X POST "http://localhost:8000/predict/train" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "period": "2y"}'
```

### 3. Make Predictions

#### Via Chat Interface
Just ask the AI assistant:
- "Predict AAPL price for next 7 days"
- "株価予測してください: トヨタ (7203.T) 10日間"
- "Forecast Tesla stock for next 2 weeks"

The AI will automatically call the prediction tool!

#### Via REST API
```bash
curl -X POST "http://localhost:8000/predict/forecast" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 7, "auto_train": true}'
```

#### Via Python
```python
from app.services.stock_prediction_service import predict_stock_price

result = predict_stock_price(
    symbol="AAPL",
    days=7,
    auto_train=True  # Auto-train if model doesn't exist
)

print(f"Current: ${result['current_price']}")
print(f"Predicted (7d): ${result['summary']['final_predicted_price']}")
print(f"Trend: {result['summary']['trend_en']}")

for pred in result['predictions']:
    print(f"{pred['date']}: ${pred['predicted_price']} ({pred['change_pct']:+.2f}%)")
```

## Model Architecture

### LSTM Network
```
Input: (60 days × 7 features)
  ↓
LSTM Layer 1: 64 units + Dropout(0.2)
  ↓
LSTM Layer 2: 32 units + Dropout(0.2)
  ↓
LSTM Layer 3: 16 units + Dropout(0.2)
  ↓
Dense Output: 1 unit (predicted close price)
```

### Training Configuration
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Mean Squared Error
- **Metrics:** Mean Absolute Error
- **Batch Size:** 32
- **Max Epochs:** 50 (with early stopping)
- **Train/Test Split:** 80/20
- **Callbacks:**
  - Early Stopping (patience=10, monitors val_loss)
  - ReduceLROnPlateau (factor=0.5, patience=5)

## GPU Optimization for GTX 1650 Ti

### Memory Growth
```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

This prevents TensorFlow from allocating all 4GB VRAM upfront.

### Model Size
- Trained model: ~10-20 MB (depends on input features)
- Scaler object: < 1 MB
- Total storage per symbol: < 25 MB

### Performance Benchmarks (GTX 1650 Ti Mobile)
- Training time: ~2-5 minutes (2 years of daily data)
- Inference time: < 1 second (7-day prediction)
- VRAM usage during training: 2-3 GB
- VRAM usage during inference: < 500 MB

## API Endpoints

### POST `/predict/forecast`
Predict future stock prices.

**Request:**
```json
{
  "symbol": "AAPL",
  "days": 7,
  "auto_train": true
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "current_price": 178.45,
  "prediction_days": 7,
  "predictions": [
    {
      "date": "2025-10-04",
      "predicted_price": 179.23,
      "change_from_current": 0.78,
      "change_pct": 0.44
    },
    ...
  ],
  "summary": {
    "final_predicted_price": 181.50,
    "total_change": 3.05,
    "total_change_pct": 1.71,
    "trend": "上昇",
    "trend_en": "bullish"
  },
  "source": "lstm_prediction"
}
```

### POST `/predict/train`
Train a new prediction model.

**Request:**
```json
{
  "symbol": "AAPL",
  "period": "2y",
  "save_model": true
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "status": "success",
  "model_path": "models/stock_predictions/AAPL_model.keras",
  "metrics": {
    "train_loss": 0.000234,
    "val_loss": 0.000456,
    "train_mae": 0.012,
    "val_mae": 0.018
  },
  "data_info": {
    "train_samples": 400,
    "test_samples": 100
  }
}
```

### GET `/predict/model/{symbol}`
Get model information.

### GET `/predict/models`
List all trained models.

## Model Storage

Models are stored in `models/stock_predictions/`:

```
models/stock_predictions/
├── AAPL_model.keras          # Trained LSTM model
├── AAPL_scaler.pkl        # Feature scaler
├── AAPL_config.json       # Model metadata
├── ^N225_model.keras
├── ^N225_scaler.pkl
└── ^N225_config.json
```

## Chat Integration

The prediction tool is automatically available in the chat interface:

**User:** "Predict Apple stock for next week"

**AI:** *[Calls predict_stock_price tool]*
```
📈 Apple (AAPL) Price Prediction

Current Price: $178.45

7-Day Forecast:
• Oct 4: $179.23 (+0.44%)
• Oct 5: $179.87 (+0.80%)
• Oct 6: $180.12 (+0.94%)
• Oct 7: $180.45 (+1.12%)
• Oct 8: $180.89 (+1.37%)
• Oct 9: $181.23 (+1.56%)
• Oct 10: $181.50 (+1.71%)

Summary: Bullish trend predicted (+$3.05, +1.71%)

⚠️ Disclaimer: This is an AI prediction for educational purposes only. Not financial advice.
```

## Supported Symbols

- **US Stocks:** AAPL, MSFT, GOOGL, AMZN, TSLA, etc.
- **Indices:** ^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow Jones)
- **Japanese Stocks:** 7203.T (Toyota), 6758.T (Sony), 9984.T (SoftBank)
- **Japanese Indices:** ^N225 (Nikkei 225), ^TPX (TOPIX)

Any Yahoo Finance ticker symbol is supported!

## Limitations & Disclaimers

⚠️ **Important:**
1. This is a **basic LSTM model** for educational/demonstration purposes
2. **Not financial advice** - do not use for real trading decisions
3. Past performance does not guarantee future results
4. Market predictions are inherently uncertain
5. Consider fundamental analysis, news, and expert opinions

### Technical Limitations
- Requires minimum 90 days of historical data for training
- Cannot predict black swan events or major market shocks
- Does not account for earnings, news, or fundamental changes
- Performance degrades for highly volatile stocks

## Advanced Usage

### Custom Training
```python
from app.services.stock_prediction_service import train_model

# Train with custom parameters
result = train_model(
    symbol="TSLA",
    period="5y",  # More data = potentially better model
    save_model=True
)

# Check performance
print(f"Validation MAE: {result['metrics']['val_mae']:.4f}")
```

### Model Performance Analysis
```python
from app.services.stock_prediction_service import get_model_info

info = get_model_info("AAPL")

if info['model_exists']:
    config = info['config']
    print(f"Trained: {config['trained_date']}")
    print(f"Val MAE: {config['val_mae']:.4f}")
    print(f"Features: {config['features']}")
```

### Batch Predictions
```bash
# Train models for multiple stocks
for symbol in AAPL MSFT GOOGL TSLA; do
    python scripts/train_prediction_model.py $symbol --period 2y
done

# List all models
python scripts/train_prediction_model.py --list
```

## Troubleshooting

### Out of Memory (OOM) Errors
If you get OOM errors on your GPU:

1. **Enable memory growth** (already enabled in code)
2. **Reduce batch size** in `MODEL_CONFIG`:
   ```python
   "batch_size": 16,  # Reduce from 32
   ```
3. **Use CPU fallback:**
   ```bash
   CUDA_VISIBLE_DEVICES="" python scripts/train_prediction_model.py AAPL
   ```

### Model Not Found
If prediction fails with "model not found":

1. **Auto-train:** Set `auto_train=True` (default)
2. **Manual train:** Run training script first
   ```bash
   python scripts/train_prediction_model.py AAPL
   ```

### Poor Predictions
If predictions are inaccurate:

1. **Use more training data:** `--period 5y`
2. **Check validation MAE:** Should be < 0.05 for good models
3. **Retrain model** with fresh data
4. **Avoid highly volatile stocks** (crypto, penny stocks)

## Future Enhancements

🚀 **Potential Improvements:**
- [ ] Transformer-based architecture (Temporal Fusion Transformer)
- [ ] Multi-task learning (predict price + volatility + trend)
- [ ] Sentiment analysis integration (news, social media)
- [ ] Ensemble models (LSTM + GRU + Attention)
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Real-time model updates (incremental learning)
- [ ] Multi-symbol correlation modeling

## References

- TensorFlow GPU Guide: https://www.tensorflow.org/guide/gpu
- LSTM for Time Series: https://www.tensorflow.org/tutorials/structured_data/time_series
- Stock Prediction Papers: arXiv:2009.10819, arXiv:2012.07436

---

**Created for:** GTX 1650 Ti Mobile (4GB VRAM)  
**Model Type:** LSTM (Long Short-Term Memory)  
**Framework:** TensorFlow 2.15.x + Keras  
**License:** Educational/Research Use Only
