# utils/backtest_lstm.py - PERBAIKAN
import os
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go


DEFAULT_MODEL_PATH = os.path.join("models", "best_lstm_model.h5")
DEFAULT_SCALER_PATH = os.path.join("models", "minmax_scaler.pkl")
DEFAULT_METADATA_PATH = os.path.join("models", "model_metadata.json")


# =====================================================
# 1) MODEL LOADER
# =====================================================
def load_model_and_scaler(model_path=None, scaler_path=None, metadata_path=None):
    mp = model_path or DEFAULT_MODEL_PATH
    sp = scaler_path or DEFAULT_SCALER_PATH
    md = metadata_path or DEFAULT_METADATA_PATH

    if not os.path.exists(mp):
        raise FileNotFoundError(f"Model not found: {mp}")
    if not os.path.exists(sp):
        raise FileNotFoundError(f"Scaler not found: {sp}")

    model = load_model(mp)

    try:
        scaler = joblib.load(sp)
    except Exception:
        import pickle
        with open(sp, "rb") as f:
            scaler = pickle.load(f)

    metadata = None
    if os.path.exists(md):
        with open(md, "r") as f:
            metadata = json.load(f)

    return model, scaler, metadata


# =====================================================
# 2) LSTM PREDICTION (FIXED - compatible with your metadata)
# =====================================================
# Cari fungsi ini di backtest_lstm.py
def predict_series_from_model(df_feat, model, scaler, meta):
    """
    Make predictions using the LSTM model.
    FIXED VERSION: Proper index handling to avoid length mismatch.
    """
    print(f"🔍 DEBUG predict_series_from_model:")
    print(f"   Input df_feat shape: {df_feat.shape}")
    print(f"   Input df_feat columns: {df_feat.columns.tolist()}")
    
    if df_feat.empty:
        print("   ❌ Empty dataframe")
        return pd.Series(dtype=float)
    
    # Ensure columns are in correct order
    if meta and 'features_used' in meta:
        expected_features = meta['features_used']
        df_feat = df_feat.reindex(columns=expected_features, fill_value=0)
    
    # Scale data
    try:
        scaled_data = scaler.transform(df_feat)
        print(f"   Scaled data shape: {scaled_data.shape}")
    except Exception as e:
        print(f"   ❌ Scaling failed: {e}")
        return pd.Series(dtype=float)
    
    # Get timesteps from metadata or use default
    timesteps = meta.get('timesteps', 10) if meta else 10
    print(f"   Using timesteps: {timesteps}")
    
    # Prepare sequences for LSTM
    X_pred = []
    for i in range(timesteps, len(scaled_data)):
        X_pred.append(scaled_data[i-timesteps:i])
    
    if not X_pred:
        print("   ❌ No data for prediction")
        return pd.Series(dtype=float)
    
    X_pred = np.array(X_pred)
    print(f"   X_pred shape for LSTM: {X_pred.shape}")
    
    # Make predictions
    try:
        predictions = model.predict(X_pred, verbose=0)
    except Exception as e:
        print(f"   ❌ Model prediction failed: {e}")
        return pd.Series(dtype=float)
    
    # Flatten predictions
    predictions = predictions.flatten()
    print(f"   Raw predictions shape: {predictions.shape}")
    
    # **FIXED: Proper index calculation**
    # Predictions start from index position 'timesteps'
    # We have len(predictions) predictions for indices starting at 'timesteps'
    
    # Expected length: len(df_feat) - timesteps
    expected_length = len(df_feat) - timesteps
    
    # Validate length
    if len(predictions) != expected_length:
        print(f"   ⚠️ WARNING: predictions length {len(predictions)} != expected {expected_length}")
        print(f"   Adjusting predictions to match expected length...")
        
        # Take the minimum of both
        actual_length = min(len(predictions), expected_length)
        predictions = predictions[:actual_length]
        print(f"   Using {actual_length} predictions")
    
    # Create index for predictions - starting from timesteps
    if timesteps < len(df_feat):
        pred_index = df_feat.index[timesteps:timesteps + len(predictions)]
    else:
        # Fallback: use last indices
        pred_index = df_feat.index[-len(predictions):] if len(predictions) <= len(df_feat) else df_feat.index
    
    # Final check: ensure lengths match
    if len(predictions) != len(pred_index):
        print(f"   ❌ CRITICAL: Final mismatch - predictions: {len(predictions)}, index: {len(pred_index)}")
        # Force alignment by taking the minimum
        min_len = min(len(predictions), len(pred_index))
        predictions = predictions[:min_len]
        pred_index = pred_index[:min_len]
    
    print(f"   ✅ Final predictions length: {len(predictions)}")
    print(f"   ✅ Index range: {pred_index[0] if len(pred_index) > 0 else 'N/A'} to {pred_index[-1] if len(pred_index) > 0 else 'N/A'}")
    
    # Create and return series
    return pd.Series(predictions, index=pred_index, name='prediction')

# =====================================================
# 3) SIGNAL GENERATOR (UPDATED - more robust)
# =====================================================
def generate_signals_from_preds(df, pred_col="prediction"):
    """
    BUY when prediction > close
    SELL when prediction < close
    Flat when equal
    """
    if pred_col not in df.columns:
        raise KeyError(f"{pred_col} not found in df!")

    df2 = df.copy()
    
    # Ensure 'close' column exists (case-insensitive)
    if 'close' not in df2.columns:
        # Try to find close column
        close_cols = [col for col in df2.columns if 'close' in col.lower()]
        if close_cols:
            df2['close'] = df2[close_cols[0]]
        else:
            raise KeyError("No 'close' column found in dataframe")
    
    df2["signal"] = 0

    cond_buy = df2[pred_col] > df2["close"]
    cond_sell = df2[pred_col] < df2["close"]

    df2.loc[cond_buy, "signal"] = 1
    df2.loc[cond_sell, "signal"] = -1

    df2["signal"] = df2["signal"].fillna(0).astype(int)
    return df2


# =====================================================
# 4) BACKTEST ENGINE (UPDATED - more robust)
# =====================================================
def run_backtest(
    df_signal: pd.DataFrame,
    initial_capital: float = 100.0,
    fee: float = 0.0,
    long_only: bool = True
):
    # Ensure required columns exist
    if "signal" not in df_signal.columns:
        raise KeyError("df_signal must contain 'signal' column.")
    
    # Find close column (case-insensitive)
    close_cols = [col for col in df_signal.columns if 'close' in col.lower()]
    if not close_cols:
        raise KeyError("No 'close' column found in dataframe")
    
    close_col = close_cols[0]
    
    df = df_signal.copy()
    df["close"] = df[close_col].astype(float).fillna(method="ffill")

    position = 0
    cash = initial_capital
    size = 0.0

    portfolio_values = []
    trades = []

    for i in range(len(df)):
        sig = df["signal"].iat[i]
        price = df["close"].iat[i]

        # BUY
        if sig == 1 and position == 0:
            size = (cash * (1 - fee)) / price
            entry_price = price
            entry_time = df.index[i]
            position = 1
            cash = 0.0

        # SELL
        elif sig == -1 and position == 1:
            cash = size * price * (1 - fee)
            pnl = cash - initial_capital
            trades.append({
                "entry_time": entry_time,
                "exit_time": df.index[i],
                "entry_price": entry_price,
                "exit_price": price,
                "pnl": pnl
            })
            size = 0
            position = 0

        # Update equity
        pv = cash + size * price
        portfolio_values.append(pv)

    # If still holding at the end
    if position == 1:
        last_price = df["close"].iat[-1]
        pv = cash + size * last_price
        portfolio_values[-1] = pv
        trades.append({
            "entry_time": entry_time,
            "exit_time": df.index[-1],
            "entry_price": entry_price,
            "exit_price": last_price,
            "pnl": pv - initial_capital
        })

    equity = pd.Series(portfolio_values, index=df.index)

    # Metrics
    if len(equity) > 0 and initial_capital > 0:
        total_return = equity.iloc[-1] / initial_capital - 1
        years = max((df.index[-1] - df.index[0]).days / 252, 1/252)
        annual_return = (1 + total_return) ** (1/years) - 1

        strat_ret = equity.pct_change().fillna(0)
        if strat_ret.std(ddof=0) > 0:
            sharpe = strat_ret.mean() / strat_ret.std(ddof=0) * np.sqrt(252)
        else:
            sharpe = None

        max_dd = (equity / equity.cummax() - 1).min()
    else:
        total_return = 0
        annual_return = 0
        sharpe = 0
        max_dd = 0

    perf = {
        "initial_capital": float(initial_capital),
        "final_capital": float(equity.iloc[-1]) if len(equity) > 0 else float(initial_capital),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe) if sharpe else None,
        "max_drawdown": float(max_dd),
        "num_trades": len(trades)
    }

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity"))

    buys = df.index[df["signal"] == 1]
    sells = df.index[df["signal"] == -1]

    if len(buys) > 0:
        fig.add_trace(go.Scatter(
            x=buys, y=df.loc[buys, "close"],
            mode="markers", marker=dict(symbol="triangle-up", color="green", size=10),
            name="BUY"
        ))
    
    if len(sells) > 0:
        fig.add_trace(go.Scatter(
            x=sells, y=df.loc[sells, "close"],
            mode="markers", marker=dict(symbol="triangle-down", color="red", size=10),
            name="SELL"
        ))

    fig.update_layout(title="Backtest: LSTM Strategy", height=500)

    return perf, fig