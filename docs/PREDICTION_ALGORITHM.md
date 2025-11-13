# Stock Price Prediction Algorithm Explained

## Overview

The system uses **LSTM (Long Short-Term Memory)** neural networks - a specialized type of Recurrent Neural Network (RNN) designed for sequential time-series data like stock prices.

## Why LSTM for Stock Prediction?

### Problem: Traditional Neural Networks Can't Remember
- Regular neural networks treat each input independently
- Stock prices depend on historical context
- Need to remember patterns over time

### Solution: LSTM Memory Cells
- **Remembers long-term patterns** (weeks/months of price trends)
- **Forgets irrelevant information** (old noise)
- **Updates memory selectively** (important events)

## Algorithm Architecture

### 1. **Input Features (7 Technical Indicators)**

```python
features = [
    "close",      # Closing price
    "volume",     # Trading volume
    "high",       # Daily high
    "low",        # Daily low
    "sma_5",      # 5-day Simple Moving Average
    "sma_20",     # 20-day Simple Moving Average
    "rsi_14"      # 14-day Relative Strength Index
]
```

**Why these features?**
- **close**: The main price to predict
- **volume**: Trading activity (high volume = strong trend)
- **high/low**: Daily price range (volatility indicator)
- **sma_5**: Short-term trend
- **sma_20**: Medium-term trend
- **rsi_14**: Momentum indicator (overbought/oversold)

### 2. **Data Preparation Pipeline**

#### Step 1: Fetch Historical Data
```python
# Get 2 years of daily stock data
data = get_historical_prices(symbol, period="2y", interval="1d")
# Result: ~500 trading days of data
```

#### Step 2: Calculate Technical Features
```python
# Simple Moving Average (SMA)
df["sma_5"] = df["close"].rolling(window=5).mean()

# Relative Strength Index (RSI)
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))
```

#### Step 3: Normalize Data (MinMaxScaler)
```python
# Scale all features to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Why? Neural networks work best with normalized inputs
# Example: $250 price → 0.75, $100 price → 0.30
```

#### Step 4: Create Sequences (Sliding Window)
```python
lookback_days = 60  # Use 60 days to predict next day

X = []  # Input sequences
y = []  # Target values

for i in range(60, len(data)):
    X.append(data[i-60:i])  # Previous 60 days
    y.append(data[i, 0])     # Next day's close price

# Result:
# X shape: (samples, 60 days, 7 features)
# y shape: (samples, 1)
```

**Example:**
```
Days 1-60   → Predict Day 61
Days 2-61   → Predict Day 62
Days 3-62   → Predict Day 63
...
```

### 3. **LSTM Neural Network Architecture**

```
Input Layer: (60 days × 7 features)
     ↓
LSTM Layer 1: 64 units + Dropout(0.2)
     ↓
LSTM Layer 2: 32 units + Dropout(0.2)
     ↓
LSTM Layer 3: 16 units + Dropout(0.2)
     ↓
Dense Output: 1 unit (predicted close price)
```

#### Layer-by-Layer Breakdown

**LSTM Layer 1 (64 units):**
- Processes 60 days of data
- 64 memory cells learn different patterns
- `return_sequences=True` → Passes full sequence to next layer
- Dropout 20% → Prevents overfitting

**LSTM Layer 2 (32 units):**
- Refines patterns from Layer 1
- 32 memory cells for intermediate features
- Still processes sequences

**LSTM Layer 3 (16 units):**
- Final pattern extraction
- 16 memory cells for high-level features
- `return_sequences=False` → Outputs single vector

**Dense Output Layer:**
- Single neuron
- Outputs predicted price (normalized 0-1)

### 4. **LSTM Cell Internal Mechanics**

Each LSTM cell has **4 gates** that control information flow:

```
┌─────────────────────────────────────┐
│         LSTM Memory Cell            │
│                                     │
│  1. Forget Gate: What to forget?   │
│     σ(Wf·[h_prev, x] + bf)         │
│                                     │
│  2. Input Gate: What to remember?   │
│     σ(Wi·[h_prev, x] + bi)         │
│                                     │
│  3. Cell State Update               │
│     tanh(Wc·[h_prev, x] + bc)      │
│                                     │
│  4. Output Gate: What to output?    │
│     σ(Wo·[h_prev, x] + bo)         │
└─────────────────────────────────────┘
```

**Real-world example:**
- **Forget Gate**: Forgets yesterday's noise
- **Input Gate**: Remembers "price crossed 20-day MA"
- **Cell State**: Stores "upward trend for 5 days"
- **Output Gate**: Outputs "predict price increase"

### 5. **Training Process**

#### Loss Function: Mean Squared Error (MSE)
```python
MSE = (1/n) * Σ(predicted_price - actual_price)²

# Why MSE?
# - Penalizes large errors more (squared)
# - Differentiable (needed for backpropagation)
# - Standard for regression problems
```

#### Optimizer: Adam
```python
Adam(learning_rate=0.001)

# Why Adam?
# - Adaptive learning rate
# - Combines momentum + RMSprop
# - Fast convergence
```

#### Training Loop
```python
for epoch in range(50):
    for batch in training_data:
        # 1. Forward pass
        predictions = model.predict(batch_X)
        
        # 2. Calculate loss
        loss = MSE(predictions, batch_y)
        
        # 3. Backpropagation
        gradients = compute_gradients(loss)
        
        # 4. Update weights
        weights = weights - learning_rate * gradients
```

#### Early Stopping
```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Stops training if validation loss doesn't improve for 10 epochs
# Prevents overfitting
```

### 6. **Prediction Algorithm**

#### Single-Step Prediction
```python
def predict_next_day(model, last_60_days):
    # 1. Scale input
    scaled_input = scaler.transform(last_60_days)
    
    # 2. Reshape for LSTM
    X = scaled_input.reshape(1, 60, 7)
    
    # 3. Predict (normalized)
    pred_scaled = model.predict(X)
    
    # 4. Inverse scale to get actual price
    pred_price = scaler.inverse_transform(pred_scaled)
    
    return pred_price
```

#### Multi-Step Prediction (7 days)
```python
def predict_7_days(model, last_60_days):
    predictions = []
    current_sequence = last_60_days.copy()
    
    for day in range(7):
        # Predict next day
        next_price = predict_next_day(model, current_sequence)
        predictions.append(next_price)
        
        # Update sequence (sliding window)
        # Remove oldest day, add predicted day
        current_sequence = np.vstack([
            current_sequence[1:],  # Drop day 1
            create_features(next_price)  # Add predicted day
        ])
    
    return predictions
```

## Mathematical Formulas

### 1. Simple Moving Average (SMA)
```
SMA_n = (P₁ + P₂ + ... + Pₙ) / n

Example (5-day SMA):
Prices: [100, 102, 101, 103, 105]
SMA_5 = (100 + 102 + 101 + 103 + 105) / 5 = 102.2
```

### 2. Relative Strength Index (RSI)
```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))

Example:
Average Gain (14 days) = 2.5
Average Loss (14 days) = 1.5
RS = 2.5 / 1.5 = 1.67
RSI = 100 - (100 / (1 + 1.67)) = 62.5

Interpretation:
RSI > 70: Overbought (may drop)
RSI < 30: Oversold (may rise)
```

### 3. LSTM Cell Equations

**Forget Gate:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Input Gate:**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

**Cell State Update:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

**Output Gate:**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

Where:
- σ = sigmoid activation (0-1)
- tanh = hyperbolic tangent (-1 to 1)
- ⊙ = element-wise multiplication
- W, b = learned weights and biases

### 4. Loss Functions

**Mean Squared Error:**
```
MSE = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²
```

**Mean Absolute Error:**
```
MAE = (1/n) Σᵢ₌₁ⁿ |yᵢ - ŷᵢ|
```

## Performance Metrics

### Training Metrics
- **Train Loss (MSE)**: ~0.016 (Good if < 0.05)
- **Val Loss (MSE)**: ~0.011 (Good if < 0.05)
- **Train MAE**: ~0.095 (Good if < 0.10)
- **Val MAE**: ~0.078 (Good if < 0.10)

### What MAE Means
```
MAE = 0.078 on normalized [0, 1] scale

If price range is $200-$300:
Actual error = 0.078 × ($300 - $200) = $7.80

Prediction: $250 ± $7.80
```

## Optimization Techniques

### 1. Dropout (Regularization)
- Randomly drops 20% of neurons during training
- Prevents overfitting to training data
- Forces network to learn robust features

### 2. Batch Normalization (Implicit)
- MinMaxScaler normalizes inputs
- Keeps gradients stable
- Faster convergence

### 3. Early Stopping
- Monitors validation loss
- Stops when overfitting starts
- Restores best weights

### 4. Learning Rate Reduction
```python
ReduceLROnPlateau(
    factor=0.5,      # Reduce by 50%
    patience=5,      # After 5 epochs
    min_lr=0.00001   # Minimum learning rate
)
```

## Limitations & Assumptions

### What the Model CAN Learn
✅ Price trends and patterns
✅ Seasonal effects
✅ Technical indicator relationships
✅ Historical volatility

### What the Model CANNOT Predict
❌ Black swan events (COVID, wars)
❌ Earnings surprises
❌ Breaking news impact
❌ Market manipulation
❌ Fundamental changes

### Assumptions
1. **Past patterns repeat**: History somewhat predicts future
2. **Market efficiency**: Prices reflect available information
3. **Stationarity**: Statistical properties don't change drastically
4. **No structural breaks**: Market regime stays similar

## Algorithm Complexity

### Time Complexity
- **Training**: O(n × e × b × u²)
  - n = data points
  - e = epochs
  - b = batches
  - u = LSTM units
  
- **Inference**: O(t × u²)
  - t = sequence length (60)
  - u = LSTM units (64, 32, 16)

### Space Complexity
- **Model Size**: O(u² × l)
  - u = units per layer
  - l = number of layers
  - ~10-20 MB for our config

### Actual Performance (GTX 1650 Ti)
- Training: 2-5 minutes (CPU), 30-60 seconds (GPU)
- Inference: < 1 second (CPU), < 100ms (GPU)

## Comparison with Other Algorithms

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **LSTM** | Handles sequences, long-term memory | Complex, slow training | Time series, stock prices |
| **ARIMA** | Simple, fast | Assumes stationarity | Stationary data |
| **Random Forest** | No sequence dependency | Can't learn time patterns | Tabular data |
| **Transformer** | Parallel processing | Needs more data | Large datasets |

## References

### Academic Papers
- Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
- Gers et al. (2000): "Learning to Forget"
- Fischer & Krauss (2018): "Deep Learning with LSTM for Financial Forecasting"

### Technical Resources
- TensorFlow LSTM Guide: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
- Stock Prediction with LSTM: arXiv:2009.10819
- Time Series Analysis: https://otexts.com/fpp3/

---

**Summary**: The system uses a 3-layer LSTM network that learns from 60 days of price history + 6 technical indicators to predict future prices. It's optimized for lightweight GPUs (4GB VRAM) and provides predictions in < 1 second after initial training.
