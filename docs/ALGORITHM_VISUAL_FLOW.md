# Stock Prediction Algorithm - Visual Flow Diagram

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER REQUEST                                  │
│  "Predict AAPL price for next 7 days"                           │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CHECK MODEL EXISTS                            │
│  Does models/AAPL_model.keras exist?                               │
└────────────┬────────────────────────────────┬───────────────────┘
             ↓ NO                              ↓ YES
┌────────────────────────────┐    ┌───────────────────────────────┐
│   AUTO-TRAIN MODEL         │    │   LOAD EXISTING MODEL         │
│   (First time only)        │    │   (Instant - reuse)           │
└────────────┬───────────────┘    └───────────┬───────────────────┘
             ↓                                  ↓
             └──────────────┬────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MAKE PREDICTION                               │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RETURN RESULT                                 │
│  "AAPL: $257.80 → $253.03 (-1.85%) over 7 days"                │
└─────────────────────────────────────────────────────────────────┘
```

## Training Process (First Time Only)

```
START: Auto-Train AAPL Model
│
├─► STEP 1: Fetch Historical Data
│   ├─ Get 2 years of AAPL daily prices (~500 days)
│   ├─ Columns: date, open, high, low, close, volume
│   └─ Source: Yahoo Finance API
│
├─► STEP 2: Calculate Technical Features
│   ├─ SMA_5  = 5-day moving average
│   ├─ SMA_20 = 20-day moving average
│   └─ RSI_14 = Relative Strength Index
│
├─► STEP 3: Normalize Data
│   ├─ Scale all features to [0, 1]
│   ├─ MinMaxScaler: (value - min) / (max - min)
│   └─ Save scaler for later inverse transform
│
├─► STEP 4: Create Sequences
│   ├─ Sliding window of 60 days
│   ├─ X: [Days 1-60] → Y: [Day 61]
│   ├─ X: [Days 2-61] → Y: [Day 62]
│   └─ Result: ~440 training samples
│
├─► STEP 5: Train/Test Split
│   ├─ Training: 80% (354 samples)
│   └─ Testing:  20% (89 samples)
│
├─► STEP 6: Build LSTM Model
│   │
│   ├─ Input: (60 days, 7 features)
│   │
│   ├─ LSTM Layer 1 (64 units)
│   │   ├─ Memory cells: 64
│   │   ├─ Dropout: 20%
│   │   └─ Output: (60, 64)
│   │
│   ├─ LSTM Layer 2 (32 units)
│   │   ├─ Memory cells: 32
│   │   ├─ Dropout: 20%
│   │   └─ Output: (60, 32)
│   │
│   ├─ LSTM Layer 3 (16 units)
│   │   ├─ Memory cells: 16
│   │   ├─ Dropout: 20%
│   │   └─ Output: (16,)
│   │
│   └─ Dense Output (1 unit)
│       └─ Output: Predicted price
│
├─► STEP 7: Train Model
│   ├─ Optimizer: Adam (lr=0.001)
│   ├─ Loss: Mean Squared Error
│   ├─ Epochs: Up to 50
│   ├─ Batch size: 32
│   ├─ Early stopping: patience=10
│   └─ Training time: 2-5 min (CPU)
│
├─► STEP 8: Evaluate Model
│   ├─ Train Loss: 0.0158
│   ├─ Val Loss:   0.0114
│   ├─ Train MAE:  0.0945
│   └─ Val MAE:    0.0780 ✓ Good!
│
└─► STEP 9: Save Model
    ├─ AAPL_model.keras     (~15 MB)
    ├─ AAPL_scaler.pkl   (~1 KB)
    └─ AAPL_config.json  (~500 bytes)

END: Model ready for predictions
```

## Prediction Process (Using Trained Model)

```
START: Predict AAPL for 7 days
│
├─► STEP 1: Load Model & Scaler
│   ├─ model  = load('AAPL_model.keras')
│   └─ scaler = load('AAPL_scaler.pkl')
│
├─► STEP 2: Get Recent Data
│   ├─ Fetch last 60 days of AAPL
│   └─ Calculate technical features
│
├─► STEP 3: Prepare Input
│   ├─ Scale data with loaded scaler
│   ├─ Shape: (1, 60, 7)
│   └─ Ready for LSTM input
│
├─► STEP 4: Predict Day 1
│   ├─ Input:  Days 1-60
│   ├─ LSTM Forward Pass
│   ├─ Output: Day 61 (scaled)
│   └─ Inverse scale: $258.45
│
├─► STEP 5: Update Sequence
│   ├─ Drop oldest day (Day 1)
│   ├─ Add predicted day (Day 61)
│   └─ New sequence: Days 2-61
│
├─► STEP 6: Predict Day 2
│   ├─ Input:  Days 2-61
│   ├─ LSTM Forward Pass
│   ├─ Output: Day 62 (scaled)
│   └─ Inverse scale: $257.89
│
├─► STEP 7: Repeat for Days 3-7
│   └─ Same process as Steps 5-6
│
└─► STEP 8: Return Results
    ├─ Day 1: $258.45 (+0.25%)
    ├─ Day 2: $257.89 (+0.04%)
    ├─ Day 3: $256.23 (-0.61%)
    ├─ Day 4: $255.12 (-1.04%)
    ├─ Day 5: $254.56 (-1.26%)
    ├─ Day 6: $253.89 (-1.52%)
    └─ Day 7: $253.03 (-1.85%)

END: 7-day forecast complete (<1 second)
```

## LSTM Cell Internal Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      LSTM MEMORY CELL                            │
│                                                                   │
│  Input: h(t-1) [previous state], x(t) [current input]           │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │  1. FORGET GATE                                     │         │
│  │     f(t) = σ(W_f · [h(t-1), x(t)] + b_f)           │         │
│  │     → Decides what to forget from cell state        │         │
│  │     → Output: 0 (forget) to 1 (keep)               │         │
│  └────────────────────────────────────────────────────┘         │
│                          ↓                                        │
│  ┌────────────────────────────────────────────────────┐         │
│  │  2. INPUT GATE                                      │         │
│  │     i(t) = σ(W_i · [h(t-1), x(t)] + b_i)           │         │
│  │     C̃(t) = tanh(W_C · [h(t-1), x(t)] + b_C)        │         │
│  │     → Decides what new info to store                │         │
│  └────────────────────────────────────────────────────┘         │
│                          ↓                                        │
│  ┌────────────────────────────────────────────────────┐         │
│  │  3. CELL STATE UPDATE                               │         │
│  │     C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)             │         │
│  │     → Combines forgotten + new info                 │         │
│  └────────────────────────────────────────────────────┘         │
│                          ↓                                        │
│  ┌────────────────────────────────────────────────────┐         │
│  │  4. OUTPUT GATE                                     │         │
│  │     o(t) = σ(W_o · [h(t-1), x(t)] + b_o)           │         │
│  │     h(t) = o(t) ⊙ tanh(C(t))                       │         │
│  │     → Decides what to output                        │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                   │
│  Output: h(t) [new hidden state], C(t) [new cell state]         │
└─────────────────────────────────────────────────────────────────┘

Real Example:
Day 1: Price $250 → Cell remembers "uptrend started"
Day 2: Price $252 → Forget gate keeps "uptrend", adds "momentum"
Day 3: Price $248 → Input gate notes "reversal", updates state
Day 4: Output gate predicts $245 based on state
```

## Data Flow Through Network

```
Input Data (Last 60 Days)
├─ Day 1:  [250.5, 1000000, 252, 248, 249.8, 247.2, 55.3]
├─ Day 2:  [251.2, 1100000, 253, 249, 250.1, 247.5, 56.8]
├─ ...
└─ Day 60: [257.8, 980000,  259, 256, 256.9, 253.4, 62.1]

              ↓ Normalize (MinMaxScaler)

Scaled Data [0-1]
├─ Day 1:  [0.745, 0.512, 0.756, 0.734, 0.741, 0.729, 0.553]
├─ Day 2:  [0.751, 0.563, 0.763, 0.738, 0.747, 0.732, 0.568]
├─ ...
└─ Day 60: [0.829, 0.501, 0.841, 0.821, 0.825, 0.814, 0.621]

              ↓ Reshape for LSTM

Input Shape: (1, 60, 7)
[Batch, Time Steps, Features]

              ↓ LSTM Layer 1 (64 units)

Hidden State 1: (1, 60, 64)
[64 learned features from 60 days]

              ↓ LSTM Layer 2 (32 units)

Hidden State 2: (1, 60, 32)
[32 refined features]

              ↓ LSTM Layer 3 (16 units)

Hidden State 3: (1, 16)
[16 high-level features, sequence collapsed]

              ↓ Dense Layer (1 unit)

Prediction (scaled): 0.815

              ↓ Inverse Scale

Final Prediction: $253.03
```

## Training Optimization Flow

```
                    START TRAINING
                          │
                          ↓
        ┌─────────────────────────────────┐
        │   Initialize Random Weights     │
        └─────────────┬───────────────────┘
                      │
        ┌─────────────▼───────────────────┐
        │   FOR EACH EPOCH (max 50)       │
        │                                  │
        │  ┌────────────────────────────┐ │
        │  │ FOR EACH BATCH (32 samples)│ │
        │  │                             │ │
        │  │  1. Forward Pass            │ │
        │  │     ├─ LSTM Layer 1         │ │
        │  │     ├─ LSTM Layer 2         │ │
        │  │     ├─ LSTM Layer 3         │ │
        │  │     └─ Dense Output         │ │
        │  │                             │ │
        │  │  2. Calculate Loss          │ │
        │  │     MSE = Σ(pred - true)²  │ │
        │  │                             │ │
        │  │  3. Backpropagation         │ │
        │  │     ├─ Compute gradients    │ │
        │  │     └─ Update weights       │ │
        │  │                             │ │
        │  │  4. Apply Dropout (20%)     │ │
        │  │     Randomly drop neurons   │ │
        │  └────────────────────────────┘ │
        │                                  │
        │  ┌────────────────────────────┐ │
        │  │ Validation Step             │ │
        │  │  - Test on holdout data     │ │
        │  │  - Calculate val_loss       │ │
        │  └────────────────────────────┘ │
        │                                  │
        │  ┌────────────────────────────┐ │
        │  │ Check Early Stopping        │ │
        │  │  If val_loss not improved   │ │
        │  │  for 10 epochs → STOP       │ │
        │  └────────────────────────────┘ │
        │                                  │
        │  ┌────────────────────────────┐ │
        │  │ Reduce Learning Rate        │ │
        │  │  If plateau → lr = lr × 0.5 │ │
        │  └────────────────────────────┘ │
        └──────────────────────────────────┘
                          │
                          ↓
        ┌─────────────────────────────────┐
        │   Save Best Model               │
        │   (lowest validation loss)      │
        └─────────────────────────────────┘
                          │
                          ↓
                    TRAINING COMPLETE
```

## Prediction Example: AAPL

```
Current: $257.80 (Oct 3, 2025)

Input Features (Last 60 Days):
┌──────┬────────┬─────────┬────────┬────────┬────────┬────────┬─────────┐
│ Day  │ Close  │ Volume  │ High   │ Low    │ SMA_5  │ SMA_20 │ RSI_14  │
├──────┼────────┼─────────┼────────┼────────┼────────┼────────┼─────────┤
│ ...  │  ...   │  ...    │  ...   │  ...   │  ...   │  ...   │  ...    │
│  56  │ 255.30 │ 1.2M    │ 256.50 │ 254.10 │ 254.80 │ 252.40 │  58.3   │
│  57  │ 256.10 │ 1.1M    │ 257.20 │ 255.00 │ 255.50 │ 252.90 │  60.1   │
│  58  │ 257.50 │ 1.3M    │ 258.40 │ 256.30 │ 256.40 │ 253.50 │  63.2   │
│  59  │ 258.20 │ 1.0M    │ 259.10 │ 257.00 │ 257.10 │ 254.10 │  64.8   │
│  60  │ 257.80 │ 0.98M   │ 259.00 │ 256.80 │ 256.90 │ 254.60 │  62.1   │
└──────┴────────┴─────────┴────────┴────────┴────────┴────────┴─────────┘

           ↓ LSTM Processing

7-Day Forecast:
┌──────┬────────────┬────────────────┬──────────────┬────────────┐
│ Day  │   Date     │  Predicted $   │  Change $    │  Change %  │
├──────┼────────────┼────────────────┼──────────────┼────────────┤
│  61  │  Oct 4     │    $258.45     │    +$0.65    │   +0.25%   │
│  62  │  Oct 7     │    $257.89     │    +$0.09    │   +0.04%   │
│  63  │  Oct 8     │    $256.23     │    -$1.57    │   -0.61%   │
│  64  │  Oct 9     │    $255.12     │    -$2.68    │   -1.04%   │
│  65  │  Oct 10    │    $254.56     │    -$3.24    │   -1.26%   │
│  66  │  Oct 11    │    $253.89     │    -$3.91    │   -1.52%   │
│  67  │  Oct 14    │    $253.03     │    -$4.77    │   -1.85%   │
└──────┴────────────┴────────────────┴──────────────┴────────────┘

Trend: Bearish (predicted decline)
Confidence: MAE = 0.078 (~$7.80 typical error)
```

---

**Key Takeaways:**
1. **LSTM remembers patterns** over 60 days
2. **7 features** capture price, volume, and momentum
3. **3-layer network** extracts hierarchical patterns
4. **Auto-trains once**, then reuses model
5. **Predictions in <1 second** after training
