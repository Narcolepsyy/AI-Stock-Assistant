"""Lightweight stock price prediction service using LSTM.

Optimized for GTX 1650 Ti Mobile (4GB VRAM):
- Small model architecture: 3-layer LSTM with 32-64 units
- Model size: < 50MB
- Inference time: < 1 second on GPU
- Training: Supports incremental learning
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json

from app.services.stock_service import get_historical_prices, _normalize_symbol
from app.core.config import TICKER_RE

logger = logging.getLogger(__name__)

# Model storage directory
MODEL_DIR = Path("models/stock_predictions")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration for GTX 1650 Ti Mobile (4GB VRAM)
MODEL_CONFIG = {
    "lstm_units": [64, 32, 16],  # 3-layer LSTM with decreasing units
    "dropout": 0.2,
    "lookback_days": 60,  # Use 60 days of history
    "prediction_days": 1,  # Predict next 1 day
    "batch_size": 32,
    "epochs": 50,
    "features": ["return"],  # Model operates on daily returns instead of raw prices
}


def _scale_sequences_with_scaler(seqs: np.ndarray, scaler: object) -> np.ndarray:
    """Scale 3D sequences of returns using a fitted scaler."""
    original_shape = seqs.shape
    scaled = scaler.transform(seqs.reshape(-1, original_shape[-1]))
    return scaled.reshape(original_shape)


def _compute_return_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return MAE/RMSE/MAPE metrics for daily returns (in percentage terms)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    base = np.where(y_true == 0, 1e-8, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / base)) * 100)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
    }


def _compute_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    neutral_tolerance: float = 0.001
) -> Dict[str, Any]:
    """Score the model's ability to predict up/down moves with a deadband for noise.

    Args:
        y_true: Actual daily returns.
        y_pred: Predicted daily returns (same scale as y_true).
        neutral_tolerance: Threshold under which returns are treated as flat.

    Returns:
        Dictionary with directional accuracy, precision/recall, and basic hit counts.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    def classify_direction(values: np.ndarray) -> np.ndarray:
        dirs = np.zeros_like(values, dtype=int)
        dirs[values > neutral_tolerance] = 1
        dirs[values < -neutral_tolerance] = -1
        return dirs

    actual_dir = classify_direction(y_true)
    pred_dir = classify_direction(y_pred)

    effective_mask = actual_dir != 0
    neutral_mask = actual_dir == 0
    pred_neutral_mask = pred_dir == 0
    up_mask = actual_dir == 1
    down_mask = actual_dir == -1
    pred_up_mask = pred_dir == 1
    pred_down_mask = pred_dir == -1

    def safe_ratio(num: int, denom: int) -> Optional[float]:
        return float(num / denom) if denom else None

    directional_hits = int(np.sum((pred_dir == actual_dir) & effective_mask))
    neutral_hits = int(np.sum(pred_neutral_mask & neutral_mask))

    return {
        "directional_accuracy": safe_ratio(directional_hits, int(effective_mask.sum())),
        "effective_samples": int(effective_mask.sum()),
        "neutral_accuracy": safe_ratio(neutral_hits, int(neutral_mask.sum())),
        "neutral_samples": int(neutral_mask.sum()),
        "up_recall": safe_ratio(int(np.sum(pred_dir[up_mask] == 1)), int(up_mask.sum())),
        "down_recall": safe_ratio(int(np.sum(pred_dir[down_mask] == -1)), int(down_mask.sum())),
        "up_precision": safe_ratio(int(np.sum(actual_dir[pred_up_mask] == 1)), int(pred_up_mask.sum())),
        "down_precision": safe_ratio(int(np.sum(actual_dir[pred_down_mask] == -1)), int(pred_down_mask.sum())),
        "class_distribution": {
            "actual_up": int(up_mask.sum()),
            "actual_down": int(down_mask.sum()),
            "actual_flat": int(neutral_mask.sum()),
            "pred_up": int(pred_up_mask.sum()),
            "pred_down": int(pred_down_mask.sum()),
            "pred_flat": int(pred_neutral_mask.sum()),
        },
        "tolerance": neutral_tolerance,
    }


def _prepare_data(
    symbol: str,
    period: str = "2y",
    lookback_days: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare univariate training sequences using daily returns."""
    # Get historical data
    hist_data = get_historical_prices(symbol, period=period, interval="1d")
    rows = hist_data.get("rows", [])

    if len(rows) < lookback_days + 30:  # Need enough data for stable training
        raise ValueError(
            f"Insufficient data: {len(rows)} days. Need at least {lookback_days + 30} days."
        )

    # Convert to DataFrame sorted by date
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close_values = df["close"].astype(float).values
    returns = pd.Series(close_values).pct_change().dropna().values

    if len(returns) <= lookback_days:
        raise ValueError(
            f"Need more than {lookback_days} daily returns to create training sequences."
        )

    sequences = []
    targets = []
    for i in range(lookback_days, len(returns)):
        sequences.append(returns[i - lookback_days:i])
        targets.append(returns[i])

    X = np.array(sequences)[..., np.newaxis]  # Add feature dimension for LSTM input
    y = np.array(targets)

    return X, y


def _build_model(input_shape: Tuple[int, int]) -> Any:
    """Build lightweight LSTM model optimized for GTX 1650 Ti Mobile.
    
    Args:
        input_shape: (lookback_days, num_features)
    
    Returns:
        Compiled Keras model
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
    
    model = keras.Sequential([
        # First LSTM layer
        layers.LSTM(
            MODEL_CONFIG["lstm_units"][0],
            return_sequences=True,
            input_shape=input_shape
        ),
        layers.Dropout(MODEL_CONFIG["dropout"]),
        
        # Second LSTM layer
        layers.LSTM(
            MODEL_CONFIG["lstm_units"][1],
            return_sequences=True
        ),
        layers.Dropout(MODEL_CONFIG["dropout"]),
        
        # Third LSTM layer
        layers.LSTM(
            MODEL_CONFIG["lstm_units"][2],
            return_sequences=False
        ),
        layers.Dropout(MODEL_CONFIG["dropout"]),
        
        # Output layer
        layers.Dense(1)
    ])
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )
    
    return model


def train_model(
    symbol: str,
    period: str = "2y",
    save_model: bool = True
) -> Dict[str, Any]:
    """Train prediction model for a stock symbol.
    
    Args:
        symbol: Stock ticker symbol
        period: Historical data period (default: 2y)
        save_model: Whether to save trained model
    
    Returns:
        Training results with metrics and model path
    """
    try:
        import tensorflow as tf
        
        # Enable GPU memory growth to avoid OOM on 4GB GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except Exception as e:
        logger.warning(f"GPU configuration failed: {e}")
    
    sym = _normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("Invalid symbol")
    
    logger.info(f"Training model for {sym} with {period} of data")
    
    # Prepare unscaled sequences and targets
    X_raw, y_raw = _prepare_data(
        sym,
        period=period,
        lookback_days=MODEL_CONFIG["lookback_days"]
    )

    # Train/test split (80/20)
    split_idx = int(len(X_raw) * 0.8)
    if split_idx == 0 or split_idx == len(X_raw):
        raise ValueError("Not enough samples to perform train/test split.")

    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

    logger.info(f"Training set: {len(X_train_raw)} samples, Test set: {len(X_test_raw)} samples")

    # Scale using training data only to avoid leakage
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_raw.reshape(-1, 1))

    X_train = _scale_sequences_with_scaler(X_train_raw, scaler)
    X_test = _scale_sequences_with_scaler(X_test_raw, scaler)
    y_train = scaler.transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_test = scaler.transform(y_test_raw.reshape(-1, 1)).reshape(-1)
    
    # Build model
    model = _build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Train model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=MODEL_CONFIG["epochs"],
        batch_size=MODEL_CONFIG["batch_size"],
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate model
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    train_mae = history.history['mean_absolute_error'][-1]
    val_mae = history.history['val_mean_absolute_error'][-1]

    # Compute metrics in return space for easier interpretation
    train_pred_scaled = model.predict(X_train, verbose=0).flatten()
    val_pred_scaled = model.predict(X_test, verbose=0).flatten()
    train_pred_returns = scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
    val_pred_returns = scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
    train_return_metrics = _compute_return_metrics(y_train_raw, train_pred_returns)
    val_return_metrics = _compute_return_metrics(y_test_raw, val_pred_returns)
    
    # Save model and scaler
    model_path = None
    scaler_path = None
    
    if save_model:
        model_path = MODEL_DIR / f"{sym}_model.keras"
        scaler_path = MODEL_DIR / f"{sym}_scaler.pkl"
        config_path = MODEL_DIR / f"{sym}_config.json"
        
        model.save(str(model_path))
        
        # Save scaler
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump({
                "symbol": sym,
                "trained_date": datetime.now().isoformat(),
                "period": period,
                "features": MODEL_CONFIG["features"],
                "lookback_days": MODEL_CONFIG["lookback_days"],
                "train_samples": len(X_train_raw),
                "test_samples": len(X_test_raw),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "train_mae_return": train_return_metrics["mae"],
                "val_mae_return": val_return_metrics["mae"],
                "train_rmse_return": train_return_metrics["rmse"],
                "val_rmse_return": val_return_metrics["rmse"],
                "train_mape_pct": train_return_metrics["mape_pct"],
                "val_mape_pct": val_return_metrics["mape_pct"],
            }, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    return {
        "symbol": sym,
        "status": "success",
        "model_path": str(model_path) if model_path else None,
        "scaler_path": str(scaler_path) if scaler_path else None,
        "metrics": {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_mae": float(train_mae),
            "val_mae": float(val_mae),
            "train_mae_return": train_return_metrics["mae"],
            "val_mae_return": val_return_metrics["mae"],
            "train_rmse_return": train_return_metrics["rmse"],
            "val_rmse_return": val_return_metrics["rmse"],
            "train_mape_pct": train_return_metrics["mape_pct"],
            "val_mape_pct": val_return_metrics["mape_pct"],
        },
        "data_info": {
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "features": MODEL_CONFIG["features"],
            "lookback_days": MODEL_CONFIG["lookback_days"],
        },
        "source": "lstm_prediction"
    }


def _load_model_and_scaler(symbol: str) -> Tuple[Any, Any]:
    """Load trained model and scaler for a symbol."""
    sym = _normalize_symbol(symbol)
    
    model_path = MODEL_DIR / f"{sym}_model.keras"
    scaler_path = MODEL_DIR / f"{sym}_scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found for {sym}. Train model first.")
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"No scaler found for {sym}. Train model first.")
    
    # Load model
    from tensorflow import keras
    model = keras.models.load_model(str(model_path))
    
    # Load scaler
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler


def predict_stock_price(
    symbol: str,
    days: int = 1,
    auto_train: bool = False
) -> Dict[str, Any]:
    """Predict future stock prices using trained LSTM model on daily returns.
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days to predict (default: 1)
        auto_train: Auto-train model if not found (default: False)
    
    Returns:
        Prediction results with forecasted prices
    """
    sym = _normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("Invalid symbol")
    
    # Check if model exists
    model_path = MODEL_DIR / f"{sym}_model.keras"
    
    if not model_path.exists():
        if auto_train:
            logger.info(f"No model found for {sym}. Auto-training...")
            train_result = train_model(sym, period="2y", save_model=True)
            logger.info(f"Auto-training complete: {train_result['metrics']}")
        else:
            raise FileNotFoundError(
                f"No trained model found for {sym}. "
                f"Train model first or set auto_train=True."
            )
    
    # Load model and scaler
    model, scaler = _load_model_and_scaler(sym)
    
    # Get recent closing prices
    lookback_days = MODEL_CONFIG["lookback_days"]
    hist_data = get_historical_prices(
        sym,
        period="3mo",  # Need enough history for the lookback window
        interval="1d"
    )
    rows = hist_data.get("rows", [])

    if len(rows) < lookback_days + 1:
        raise ValueError(f"Insufficient recent data: {len(rows)} days. Need {lookback_days + 1} days.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close_values = df["close"].astype(float).values
    returns = pd.Series(close_values).pct_change().dropna().values

    if len(returns) < lookback_days:
        raise ValueError(
            f"Insufficient recent returns: have {len(returns)} but need {lookback_days} days of returns."
        )

    last_sequence_raw = returns[-lookback_days:].reshape(-1, 1)
    current_sequence = scaler.transform(last_sequence_raw)

    prediction_dates = []
    predicted_returns = []
    predicted_prices = []

    last_date = df["date"].iloc[-1]
    last_price = float(close_values[-1])

    for _ in range(days):
        X_pred = current_sequence.reshape(1, lookback_days, 1)
        pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
        pred_return = float(scaler.inverse_transform([[pred_scaled]])[0, 0])
        predicted_returns.append(pred_return)

        # Convert return to price for reporting
        next_price = last_price * (1 + pred_return)
        predicted_prices.append(float(next_price))
        last_price = next_price

        # Advance to the next trading day
        next_date = last_date + timedelta(days=1)
        while next_date.weekday() >= 5:  # Skip weekends
            next_date += timedelta(days=1)
        prediction_dates.append(next_date.strftime("%Y-%m-%d"))
        last_date = next_date

        # Feed prediction back into the sequence (in scaled return space)
        current_sequence = np.vstack([current_sequence[1:], [[pred_scaled]]])
    
    # Get current price for comparison
    current_price = float(df["close"].iloc[-1])
    
    # Calculate prediction change
    final_predicted_price = float(predicted_prices[-1])
    price_change = final_predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Determine trend
    trend = "上昇" if price_change > 0 else "下落" if price_change < 0 else "横ばい"
    trend_en = "bullish" if price_change > 0 else "bearish" if price_change < 0 else "neutral"
    
    return {
        "symbol": sym,
        "current_price": round(current_price, 2),
        "prediction_days": days,
        "predictions": [
            {
                "date": date,
                "predicted_price": round(price, 2),
                "change_from_current": round(price - current_price, 2),
                "change_pct": round(((price - current_price) / current_price) * 100, 2),
                "predicted_return_pct": round(ret * 100, 2),
            }
            for date, price, ret in zip(prediction_dates, predicted_prices, predicted_returns)
        ],
        "summary": {
            "final_predicted_price": round(final_predicted_price, 2),
            "total_change": round(price_change, 2),
            "total_change_pct": round(price_change_pct, 2),
            "trend": trend,
            "trend_en": trend_en,
        },
        "model_info": {
            "lookback_days": lookback_days,
            "features_used": MODEL_CONFIG["features"],
        },
        "source": "lstm_prediction",
        "disclaimer": "This is an AI prediction for educational purposes only. Not financial advice."
    }


def evaluate_model_accuracy(symbol: str, period: str = "6mo") -> Dict[str, Any]:
    """Evaluate a trained model on historical daily returns to verify accuracy."""
    sym = _normalize_symbol(symbol)
    if not sym or not TICKER_RE.match(sym):
        raise ValueError("Invalid symbol")

    model, scaler = _load_model_and_scaler(sym)

    X_raw, y_raw = _prepare_data(
        sym,
        period=period,
        lookback_days=MODEL_CONFIG["lookback_days"]
    )

    if len(X_raw) == 0:
        raise ValueError("Not enough data to evaluate accuracy.")

    X_scaled = _scale_sequences_with_scaler(X_raw, scaler)
    preds_scaled = model.predict(X_scaled, verbose=0).flatten()
    preds_returns = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    metrics = _compute_return_metrics(y_raw, preds_returns)
    directional_metrics = _compute_directional_accuracy(y_raw, preds_returns)

    recent_examples = [
        {
            "actual_return_pct": round(float(actual) * 100, 2),
            "predicted_return_pct": round(float(pred) * 100, 2),
            "error_pct": round(float((pred - actual) * 100), 2),
        }
        for actual, pred in zip(y_raw[-5:], preds_returns[-5:])
    ]

    return {
        "symbol": sym,
        "period": period,
        "samples": len(y_raw),
        "metrics": {
            "mae_return": metrics["mae"],
            "rmse_return": metrics["rmse"],
            "mape_pct": metrics["mape_pct"],
        },
        "directional_metrics": directional_metrics,
        "recent_examples": recent_examples,
        "source": "lstm_prediction",
    }


def get_model_info(symbol: str) -> Dict[str, Any]:
    """Get information about trained model for a symbol."""
    sym = _normalize_symbol(symbol)
    
    config_path = MODEL_DIR / f"{sym}_config.json"
    
    if not config_path.exists():
        return {
            "symbol": sym,
            "model_exists": False,
            "message": f"No trained model found for {sym}"
        }
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return {
        "symbol": sym,
        "model_exists": True,
        "config": config,
        "source": "lstm_prediction"
    }


def list_available_models() -> Dict[str, Any]:
    """List all available trained models."""
    models = []
    
    for config_file in MODEL_DIR.glob("*_config.json"):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                models.append({
                    "symbol": config.get("symbol"),
                    "trained_date": config.get("trained_date"),
                    "val_mae": config.get("val_mae"),
                })
        except Exception as e:
            logger.warning(f"Failed to read {config_file}: {e}")
            continue
    
    return {
        "count": len(models),
        "models": sorted(models, key=lambda x: x.get("trained_date", ""), reverse=True),
        "source": "lstm_prediction"
    }
