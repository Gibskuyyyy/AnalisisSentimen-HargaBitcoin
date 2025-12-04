# Di bagian imports file utama (Price.py)
import sys
import os

# Tambahkan root project ke PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
utils_path = os.path.join(project_root, "utils")

if utils_path not in sys.path:
    sys.path.append(utils_path)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px

from utils.influx_helper import get_latest_forecast
from utils.indicators import add_indicators, calculate_all_indicators
from utils.signal_screener import add_screener_signals
from utils.divergence import detect_divergences
from utils.volume_profile import compute_volume_profile
from utils.market_regime import detect_market_regime
from utils.support_resistance import find_support_resistance
from utils.backtest_lstm import (
    load_model_and_scaler,
    predict_series_from_model,
    generate_signals_from_preds,
    run_backtest,
)
from utils.orderbook import fetch_binance_orderbook
from utils.feature_builder import build_features_for_model  # Tambah ini


# Error handler global
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions."""
    print("💥 UNCAUGHT EXCEPTION:", file=sys.stderr)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

sys.excepthook = handle_exception

def debug_scaler_issue(df_feat, scaler, meta):
    """Debug issue dengan scaler yang mengharapkan 24 fitur."""
    print(f"\n{'='*60}")
    print("DEBUG SCALER ISSUE")
    print(f"{'='*60}")
    
    print(f"df_feat shape: {df_feat.shape}")
    print(f"df_feat columns: {df_feat.columns.tolist()}")
    
    if scaler:
        print(f"Scaler expects: {scaler.n_features_in_} features")
    
    if meta and 'features_used' in meta:
        print(f"Metadata expects: {len(meta['features_used'])} features")
        print(f"Metadata features: {meta['features_used']}")
        
        # Cari perbedaan
        expected_set = set(meta['features_used'])
        actual_set = set(df_feat.columns)
        
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        
        if missing:
            print(f"❌ Missing columns: {list(missing)}")
        
        if extra:
            print(f"⚠️ Extra columns: {list(extra)}")
    
    # Cek tipe data
    print(f"\nData types:")
    for col in df_feat.columns:
        print(f"  {col}: {df_feat[col].dtype}")
    
    # Cek nilai pertama
    if len(df_feat) > 0:
        print(f"\nFirst row values:")
        for col in df_feat.columns:
            print(f"  {col}: {df_feat[col].iloc[0]}")

# Juga tambahkan ini untuk menangkap warning
import warnings
warnings.filterwarnings('ignore')

def simple_predict_fix(df_feat, model, scaler, meta):
    """Simple prediction function with proper inverse scaling."""
    try:
        # Scale data
        scaled = scaler.transform(df_feat)
        
        # Get timesteps - PAKAI window_size dari metadata!
        timesteps = meta.get('best_parameters', {}).get('window_size', 
                   meta.get('timesteps', 10))
        print(f"DEBUG: Using window_size/timesteps: {timesteps}")        
        # Prepare sequences
        X = []
        for i in range(timesteps, len(scaled)):
            X.append(scaled[i-timesteps:i])
        
        if not X:
            print("DEBUG: No sequences created")
            return pd.Series(dtype=float)
        
        X = np.array(X)
        print(f"DEBUG: X shape for LSTM: {X.shape}")
        
        # Predict
        preds_scaled = model.predict(X, verbose=0).flatten()
        print(f"DEBUG: Raw predictions (scaled) - First 5: {preds_scaled[:5]}")
        print(f"DEBUG: Raw predictions range: [{preds_scaled.min():.4f}, {preds_scaled.max():.4f}]")
        
        # **CRITICAL FIX: Inverse transform predictions**
        # Kita perlu mengembalikan prediksi ke skala asli
        print(f"DEBUG: Attempting inverse transform...")
        
        # Cara 1: Inverse transform dengan membuat data dummy
        # Karena scaler mengharapkan 24 fitur, kita buat array dengan semua fitur
        n_features = scaler.n_features_in_
        n_preds = len(preds_scaled)
        
        # Buat array dummy dengan semua fitur (isi dengan 0)
        dummy = np.zeros((n_preds, n_features))
        
        # Cari index kolom 'close' dalam features
        if meta and 'features_used' in meta:
            features = meta['features_used']
            if 'close' in features:
                close_idx = features.index('close')
            elif 'Close' in features:
                close_idx = features.index('Close')
            else:
                # Coba cari dengan case-insensitive
                close_idx = None
                for i, f in enumerate(features):
                    if 'close' in f.lower():
                        close_idx = i
                        break
                
                if close_idx is None:
                    close_idx = 3  # Asumsi: 'Close' adalah kolom ke-3 (0:Open, 1:High, 2:Low, 3:Close)
        
        # Isi kolom close dengan prediksi scaled
        dummy[:, close_idx] = preds_scaled
        
        # Inverse transform
        dummy_inversed = scaler.inverse_transform(dummy)
        
        # Ambil kolom close yang sudah di-inverse
        preds = dummy_inversed[:, close_idx]
        
        print(f"DEBUG: After inverse - First 5: {preds[:5]}")
        print(f"DEBUG: After inverse range: [{preds.min():.2f}, {preds.max():.2f}]")
        
        # Create index
        start_idx = timesteps
        end_idx = start_idx + len(preds)
        
        if end_idx > len(df_feat):
            preds = preds[:len(df_feat) - start_idx]
            end_idx = len(df_feat)
        
        pred_index = df_feat.index[start_idx:end_idx]
        
        return pd.Series(preds, index=pred_index, name='prediction')
    
    except Exception as e:
        print(f"Simple predict error: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series(dtype=float)
    
# --------------------------------------------------
# Utility helpers (defensive)
# --------------------------------------------------
def clean_duplicate_columns(df):
    """Hapus kolom duplikat, simpan yang pertama."""
    if isinstance(df, pd.DataFrame):
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def safe_convert_to_numeric(data):
    """Convert data to numeric Series safely."""
    try:
        # Jika ini DataFrame, ambil kolom pertama
        if isinstance(data, pd.DataFrame):
            if data.shape[1] > 0:
                data = data.iloc[:, 0]
            else:
                return pd.Series(np.nan)
        
        # Jika ini ndarray, convert ke Series
        elif isinstance(data, np.ndarray):
            data = pd.Series(data.ravel())
        
        # Jika bukan Series, buat Series
        elif not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        # Convert ke numeric
        return pd.to_numeric(data, errors='coerce')
    
    except Exception as e:
        print(f"Error in safe_convert_to_numeric: {e}")
        return pd.Series(np.nan)

# ... fungsi-fungsi utility lainnya ...

def safe_add_indicators(df):
    """Wrapper untuk add_indicators dengan penanganan error yang lebih baik."""
    try:
        df = df.copy()
        df = clean_duplicate_columns(df)
        
        print(f"safe_add_indicators - Input shape: {df.shape}")
        print(f"safe_add_indicators - Input columns: {df.columns.tolist()}")
        
        # Pastikan kolom OHLCV ada dengan nama KAPITAL
        kapital_cols = ["Open", "High", "Low", "Close", "Volume"]
        
        # Hapus duplikat lowercase jika ada
        cols_to_keep = []
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in ['open', 'high', 'low', 'close', 'volume']:
                # Jika ini lowercase dan kapital sudah ada, skip
                kapital_version = col_lower.capitalize()
                if kapital_version in df.columns and col != kapital_version:
                    print(f"Dropping lowercase duplicate: {col}")
                    continue
            cols_to_keep.append(col)
        
        df = df[cols_to_keep]
        df = clean_duplicate_columns(df)
        
        # Pastikan semua kolom kapital ada
        for col in kapital_cols:
            if col not in df.columns:
                # Cari lowercase version
                lower_col = col.lower()
                if lower_col in df.columns:
                    df[col] = df[lower_col]
                    print(f"Copied {lower_col} to {col}")
                else:
                    print(f"Creating missing column: {col} with NaN")
                    df[col] = np.nan
        
        # Konversi ke numeric dengan aman - PERBAIKAN DI SINI
        for col in kapital_cols:
            if col in df.columns:
                # Gunakan safe_convert_to_numeric yang sudah didefinisikan
                df[col] = safe_convert_to_numeric(df[col])
        
        # Sekarang panggil add_indicators
        print(f"Calling add_indicators with columns: {[c for c in kapital_cols if c in df.columns]}")
        result = add_indicators(df)
        
        # Tambahkan kolom lowercase untuk konsistensi
        for kapital, lower in zip(kapital_cols, ['open', 'high', 'low', 'close', 'volume']):
            if kapital in result.columns and lower not in result.columns:
                result[lower] = result[kapital]
        
        result = clean_duplicate_columns(result)
        print(f"safe_add_indicators - Success! Output shape: {result.shape}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in safe_add_indicators: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: return DataFrame minimal
        df_fallback = clean_duplicate_columns(df.copy())
        for col in ['RSI_14', 'MACD', 'BB_low', 'BB_high', 'SMA_10', 'EMA_12']:
            if col not in df_fallback.columns:
                df_fallback[col] = np.nan
        
        return df_fallback

def predict_with_influx_style(df, model, scaler, meta):
    """
    Prediksi LSTM yang sama persis untuk Basic dan Pro mode
    Menggunakan logika yang identik dengan update_prediction.py
    """
    try:
        # Debug info
        print(f"\n{'='*60}")
        print("LSTM PREDICTION (SYNCED FOR BOTH MODES)")
        print(f"{'='*60}")
        print(f"Input df shape: {df.shape}")
        print(f"Model loaded: {model is not None}")
        print(f"Scaler features in: {scaler.n_features_in_ if scaler else 'N/A'}")
        
        # 1. Bangun fitur PERSIS seperti di update_prediction.py
        df_features = build_features_for_model(df.copy(), meta)
        
        if df_features.empty:
            print("❌ Feature DataFrame kosong")
            return pd.Series(dtype=float)
        
        print(f"✅ Features shape: {df_features.shape}")
        print(f"Features columns: {df_features.columns.tolist()}")
        
        # 2. Scale data
        try:
            scaled = scaler.transform(df_features)
            print(f"✅ Data scaled: {scaled.shape}")
        except Exception as e:
            print(f"❌ Scaling failed: {e}")
            # Coba dengan fitur yang sesuai
            expected_features = meta.get('features_used', [])
            missing = [f for f in expected_features if f not in df_features.columns]
            if missing:
                print(f"Missing features: {missing}")
                # Tambahkan fitur yang hilang dengan 0
                for f in missing:
                    df_features[f] = 0
                df_features = df_features[expected_features]
                scaled = scaler.transform(df_features)
                print(f"✅ Scaling after fix: {scaled.shape}")
            else:
                raise e
        
        # 3. Ambil window_size dari metadata
        window_size = meta.get('best_parameters', {}).get('window_size', 
                     meta.get('timesteps', 10))
        print(f"✅ Window size: {window_size}")
        
        # 4. Pastikan data cukup untuk window_size
        if len(scaled) < window_size + 1:
            print(f"❌ Insufficient data: {len(scaled)} < {window_size + 1}")
            return pd.Series(dtype=float)
        
        # 5. Buat sequences untuk LSTM
        X = []
        for i in range(window_size, len(scaled)):
            X.append(scaled[i-window_size:i])
        
        if not X:
            print("❌ No sequences created")
            return pd.Series(dtype=float)
        
        X = np.array(X)
        print(f"✅ Input shape for LSTM: {X.shape}")
        
        # 6. Prediksi
        print("🔄 Making predictions...")
        preds_scaled = model.predict(X, verbose=0).flatten()
        print(f"✅ Raw predictions (scaled): {preds_scaled[:5]}...")
        print(f"✅ Predictions count: {len(preds_scaled)}")
        
        # 7. INVERSE TRANSFORM - PENTING!
        n_features = scaler.n_features_in_
        n_preds = len(preds_scaled)
        
        # Buat array dummy dengan semua fitur
        dummy = np.zeros((n_preds, n_features))
        
        # Cari index fitur 'close' dalam metadata
        features_used = meta.get('features_used', [])
        close_idx = 3  # default: biasanya ke-3 (0:Open, 1:High, 2:Low, 3:Close)
        
        # Cari kolom 'close' (case-insensitive)
        for i, feat in enumerate(features_used):
            if isinstance(feat, str) and 'close' in feat.lower():
                close_idx = i
                print(f"✅ 'close' column found at index {i}: {feat}")
                break
        
        # Isi kolom close dengan prediksi scaled
        dummy[:, close_idx] = preds_scaled
        
        # Inverse transform
        dummy_inversed = scaler.inverse_transform(dummy)
        
        # Ambil kolom close yang sudah di-inverse
        preds = dummy_inversed[:, close_idx]
        
        print(f"✅ Predictions after inverse transform:")
        print(f"   Min: {preds.min():.2f}, Max: {preds.max():.2f}")
        print(f"   First 5: {preds[:5]}")
        print(f"   Last 5: {preds[-5:]}")
        
        # 8. Buat index untuk prediksi (sesuai dengan data asli)
        pred_start_idx = window_size
        pred_end_idx = pred_start_idx + len(preds)
        
        if pred_end_idx > len(df_features):
            preds = preds[:len(df_features) - pred_start_idx]
            pred_end_idx = len(df_features)
        
        pred_index = df_features.index[pred_start_idx:pred_end_idx]
        
        # 9. Buat Series dengan nama yang konsisten
        result = pd.Series(preds, index=pred_index, name='LSTM_Prediction')
        
        print(f"✅ Final prediction series: {len(result)} points")
        print(f"   Index range: {result.index[0]} to {result.index[-1]}")
        print(f"{'='*60}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in predict_with_influx_style: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series(dtype=float)

def ensure_columns(df: pd.DataFrame, cols):
    """Ensure df contains cols; if missing create as NaN series aligned to df.index."""
    df = clean_duplicate_columns(df)
    for c in cols:
        if c not in df.columns:
            # Coba cari dengan variasi nama (case-insensitive)
            found = False
            for variation in [c.lower(), c.upper(), c.capitalize(), c.title()]:
                if variation in df.columns:
                    df[c] = df[variation]
                    found = True
                    break
            
            if not found:
                df[c] = pd.Series(np.nan, index=df.index)
    return clean_duplicate_columns(df)

def compute_features_update_style(df_raw):
    """
    Hitung indikator persis seperti di update_prediction.py
    """
    import ta
    
    df_ind = df_raw.copy()
    
    # Pastikan kolom OHLCV ada
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df_ind.columns:
            raise ValueError(f"Kolom {col} tidak ditemukan")
    
    # Flatten series jika perlu
    close = pd.Series(df_ind['Close'].values.flatten(), index=df_ind.index)
    high = pd.Series(df_ind['High'].values.flatten(), index=df_ind.index)
    low = pd.Series(df_ind['Low'].values.flatten(), index=df_ind.index)
    volume = pd.Series(df_ind['Volume'].values.flatten(), index=df_ind.index)
    
    # Moving averages
    df_ind['SMA_10'] = ta.trend.sma_indicator(close, window=10)
    df_ind['SMA_50'] = ta.trend.sma_indicator(close, window=50)
    df_ind['EMA_12'] = ta.trend.ema_indicator(close, window=12)
    df_ind['EMA_26'] = ta.trend.ema_indicator(close, window=26)
    
    # Momentum
    df_ind['RSI_14'] = ta.momentum.rsi(close, window=14)
    
    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df_ind['MACD'] = macd.macd()
    df_ind['MACD_signal'] = macd.macd_signal()
    df_ind['MACD_hist'] = df_ind['MACD'] - df_ind['MACD_signal']
    
    # Volatility
    df_ind['ATR_14'] = ta.volatility.average_true_range(high, low, close, window=14)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df_ind['BB_middle'] = bb.bollinger_mavg()
    df_ind['BB_high'] = bb.bollinger_hband()
    df_ind['BB_low'] = bb.bollinger_lband()
    df_ind['BB_width'] = (df_ind['BB_high'] - df_ind['BB_low']) / df_ind['BB_middle']
    
    # On-Balance Volume
    df_ind['OBV'] = ta.volume.on_balance_volume(close, volume)
    df_ind['OBV_chg'] = df_ind['OBV'].pct_change()
    
    # Stochastic Oscillator
    df_ind['STOCH'] = ta.momentum.stoch(high, low, close, window=14, smooth_window=3)
    
    # Williams %R
    df_ind['WILLR'] = ta.momentum.williams_r(high, low, close, lbp=14)
    
    # Returns
    df_ind['ret_1d'] = close.pct_change()
    df_ind['log_ret_1d'] = np.log(close).diff()
    
    # Fill NaN (tapi jangan drop seperti di update_prediction.py)
    df_ind = df_ind.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df_ind

def build_features_for_model(df, meta):
    """
    Membangun fitur persis seperti di update_prediction.py
    """
    if meta is None or "features_used" not in meta:
        raise ValueError("metadata.json tidak memiliki daftar fitur!")

    needed = meta["features_used"]
    print(f"🚀 Features needed by model ({len(needed)}): {needed}")
    
    # Buat salinan dataframe
    df_tmp = df.copy()
    df_tmp = clean_duplicate_columns(df_tmp)
    
    # Pastikan kolom OHLCV dengan nama KAPITAL ada
    capital_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in capital_cols:
        if col not in df_tmp.columns:
            # Cari lowercase version
            lower_col = col.lower()
            if lower_col in df_tmp.columns:
                df_tmp[col] = df_tmp[lower_col]
            else:
                df_tmp[col] = np.nan
    
    # **HITUNG INDIKATOR PERSIS seperti update_prediction.py**
    print("🔧 Calculating indicators (update_prediction.py style)...")
    df_features = compute_features_update_style(df_tmp)
    
    # Pastikan semua kolom yang diperlukan ada
    df_feat = pd.DataFrame(index=df_features.index)
    
    for feature in needed:
        if feature in df_features.columns:
            df_feat[feature] = df_features[feature]
        else:
            # Cari case-insensitive
            found = False
            for col in df_features.columns:
                if col.lower() == feature.lower():
                    df_feat[feature] = df_features[col]
                    found = True
                    print(f"📝 Using {col} for {feature} (case-insensitive)")
                    break
            if not found:
                print(f"⚠️ Feature '{feature}' not found, creating with 0")
                df_feat[feature] = 0
    
    # Urutkan kolom sesuai metadata
    df_feat = df_feat.reindex(columns=needed, fill_value=0)
    
    # Fill NaN
    df_feat = df_feat.ffill().bfill().fillna(0)
    
    print(f"✅ Final feature shape: {df_feat.shape}")
    print(f"✅ Feature columns: {df_feat.columns.tolist()}")
    
    return df_feat

def force_close_column(df):
    """Memastikan kolom 'close' (lowercase) ada di DataFrame."""
    df = df.copy()
    
    # Jika 'close' sudah ada, return
    if 'close' in df.columns:
        return df
    
    # Cari variasi 'close'
    close_variants = [col for col in df.columns if 'close' in col.lower()]
    
    if close_variants:
        # Gunakan yang pertama
        df['close'] = df[close_variants[0]]
        print(f"force_close_column: Using {close_variants[0]} as 'close'")
    elif 'Close' in df.columns:
        df['close'] = df['Close']
        print(f"force_close_column: Using 'Close' as 'close'")
    else:
        # Cari kolom numeric lainnya
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df['close'] = df[numeric_cols[0]]
            print(f"force_close_column: Using {numeric_cols[0]} as 'close'")
        else:
            df['close'] = 0
            print(f"force_close_column: Creating 'close' with 0")
    
    return df

def safe_last_float(series_or_val, default=np.nan):
    """Return last value as float (safe)."""
    try:
        if isinstance(series_or_val, (pd.Series, pd.DataFrame)):
            # If DataFrame, take first column then last element
            if isinstance(series_or_val, pd.DataFrame):
                s = series_or_val.iloc[:, 0]
            else:
                s = series_or_val
            val = s.iloc[-1]
        else:
            val = series_or_val
        return float(np.array(val).reshape(-1)[-1])
    except Exception:
        return default

def check_metadata_features(meta):
    """Cek fitur dalam metadata dan cari 'close'."""
    if not meta or 'features_used' not in meta:
        return "No metadata or features_used found"
    
    features = meta['features_used']
    close_features = [f for f in features if 'close' in f.lower()]
    
    result = f"Total features: {len(features)}\n"
    result += f"Features with 'close': {close_features}\n"
    
    # Cek case sensitivity
    lower_features = [f for f in features if f.islower()]
    upper_features = [f for f in features if f.isupper()]
    title_features = [f for f in features if f.istitle()]
    
    result += f"\nCase analysis:\n"
    result += f"  Lowercase: {len(lower_features)}\n"
    result += f"  Uppercase: {len(upper_features)}\n"
    result += f"  Title case: {len(title_features)}\n"
    
    return result

# Tambahkan di sidebar untuk debugging

def safe_call_screener(df, rsi_buy, rsi_sell):
    """Call add_screener_signals but fallback to simple rule if it fails."""
    try:
        out = add_screener_signals(df, rsi_buy=rsi_buy, rsi_sell=rsi_sell)
        # Validate returned object
        if out is None or not isinstance(out, pd.DataFrame):
            raise ValueError("screener returned non-DataFrame")
        # Ensure signals exist
        if 'signal_buy' not in out.columns or 'signal_sell' not in out.columns:
            raise ValueError("screener missing expected columns")
        return out
    except Exception as e:
        # Fallback: build simple signals using numpy (avoid pandas alignment problems)
        df2 = df.copy()
        ensure_columns(df2, ['close','RSI_14','BB_low','BB_high'])
        df2 = clean_duplicate_columns(df2)
        
        idx = df2.index
        n = len(idx)
        # Coerce to numpy arrays
        close = np.asarray(df2['close'].astype(float).fillna(np.nan))
        rsi = np.asarray(df2['RSI_14'].astype(float).fillna(np.nan))
        bb_low = np.asarray(df2['BB_low'].astype(float).fillna(np.nan))
        bb_high = np.asarray(df2['BB_high'].astype(float).fillna(np.nan))

        def lt_safe(a,b):
            with np.errstate(invalid='ignore'):
                res = a < b
            res = np.where(np.isnan(res), False, res)
            return res

        def gt_safe(a,b):
            with np.errstate(invalid='ignore'):
                res = a > b
            res = np.where(np.isnan(res), False, res)
            return res

        buy = np.logical_or(lt_safe(rsi, rsi_buy), lt_safe(close, bb_low))
        sell = np.logical_or(gt_safe(rsi, rsi_sell), gt_safe(close, bb_high))

        df2['signal_buy'] = pd.Series(buy.astype(bool), index=idx)
        df2['signal_sell'] = pd.Series(sell.astype(bool), index=idx)
        df2 = clean_duplicate_columns(df2)
        return df2

def safe_detect_divergences(df, lookback, indicator):
    """Call detect_divergences and ensure return is DataFrame or None."""
    try:
        df_clean = clean_duplicate_columns(df.copy())
        out = detect_divergences(df_clean, lookback=lookback, indicator=indicator)
        if out is None or not isinstance(out, pd.DataFrame):
            return None
        return clean_duplicate_columns(out)
    except Exception:
        return None

def safe_find_support_resistance(df, n=6):
    """Call find_support_resistance and normalize its output to lists of (time,price)."""
    try:
        df_clean = clean_duplicate_columns(df.copy())
        sup, res = find_support_resistance(df_clean, n=n)
        
        # Accept several return shapes: arrays of prices, arrays of indices, or list of tuples
        # If sup/res are arrays of prices -> convert to list of (index, price) by matching closest indices
        def to_list(levels):
            if levels is None:
                return []
            # If list of tuples already
            if isinstance(levels, list) and len(levels)>0 and isinstance(levels[0], (list, tuple)) and len(levels[0])>=2:
                return [(pd.to_datetime(x[0]), float(x[1])) for x in levels]
            # If numpy array of prices -> find nearest index occurrences in df['close']
            try:
                arr = np.asarray(levels).astype(float).reshape(-1)
                out = []
                if len(arr)==0:
                    return out
                # For each price, find closest index timestamp where close is near that value
                closes = np.asarray(df_clean['close'].astype(float).fillna(np.nan))
                for v in arr:
                    if np.isnan(v):
                        continue
                    # find index of closest
                    idx = int(np.nanargmin(np.abs(closes - v)))
                    ts = df_clean.index[idx]
                    out.append((ts, float(v)))
                return out
            except Exception:
                return []
        sup_list = to_list(sup)
        res_list = to_list(res)
        return sup_list, res_list
    except Exception:
        return [], []

def run_simple_backtest(df, initial_capital=1000.0, fee=0.001, stop_loss_pct=0.05, take_profit_pct=0.10):
    """
    Simple backtest function yang lebih reliable
    """
    try:
        # Pastikan kolom yang diperlukan ada
        required_cols = ['close', 'signal']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
        
        # Buat salinan dataframe
        df_bt = df.copy()
        df_bt = clean_duplicate_columns(df_bt)
        
        # Inisialisasi variabel trading
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long position
        entry_price = 0
        entry_idx = None
        equity_curve = []
        trades = []
        
        # Iterasi melalui dataframe
        for i in range(1, len(df_bt)):
            current_idx = df_bt.index[i]
            current_price = float(df_bt['close'].iloc[i])
            prev_signal = int(df_bt['signal'].iloc[i-1])  # Gunakan sinyal sebelumnya untuk aksi hari ini
            
            # Check stop loss dan take profit jika ada posisi
            if position == 1 and entry_price > 0:
                # Hitung PnL
                current_pnl_pct = (current_price - entry_price) / entry_price
                
                # Stop Loss
                if current_pnl_pct <= -stop_loss_pct:
                    # Exit karena stop loss
                    exit_price = entry_price * (1 - stop_loss_pct)
                    pnl = (exit_price - entry_price) / entry_price * 100
                    
                    trades.append({
                        'entry_time': entry_idx,
                        'exit_time': current_idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'reason': 'Stop Loss'
                    })
                    
                    # Update capital
                    capital = capital * (1 + pnl/100 - fee*2)  # Fee untuk entry dan exit
                    position = 0
                    entry_price = 0
                
                # Take Profit
                elif current_pnl_pct >= take_profit_pct:
                    # Exit karena take profit
                    exit_price = entry_price * (1 + take_profit_pct)
                    pnl = (exit_price - entry_price) / entry_price * 100
                    
                    trades.append({
                        'entry_time': entry_idx,
                        'exit_time': current_idx,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl,
                        'reason': 'Take Profit'
                    })
                    
                    # Update capital
                    capital = capital * (1 + pnl/100 - fee*2)
                    position = 0
                    entry_price = 0
            
            # Trading signals (hanya jika tidak ada posisi)
            if position == 0:
                # Buy signal
                if prev_signal == 1:
                    position = 1
                    entry_price = current_price
                    entry_idx = current_idx
                
                # Sell signal (short jika diizinkan)
                elif prev_signal == -1:
                    # Untuk simplicity, kita skip short selling dulu
                    pass
            
            # Exit signal jika ada posisi
            elif position == 1 and prev_signal == -1:
                # Exit long position
                pnl = (current_price - entry_price) / entry_price * 100
                
                trades.append({
                    'entry_time': entry_idx,
                    'exit_time': current_idx,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': pnl,
                    'reason': 'Signal Exit'
                })
                
                # Update capital
                capital = capital * (1 + pnl/100 - fee*2)
                position = 0
                entry_price = 0
            
            # Record equity
            equity_curve.append(capital)
        
        # Close any open position at the end
        if position == 1 and entry_price > 0:
            last_price = float(df_bt['close'].iloc[-1])
            pnl = (last_price - entry_price) / entry_price * 100
            
            trades.append({
                'entry_time': entry_idx,
                'exit_time': df_bt.index[-1],
                'entry_price': entry_price,
                'exit_price': last_price,
                'pnl_pct': pnl,
                'reason': 'End of Period'
            })
            
            capital = capital * (1 + pnl/100 - fee*2)
        
        # Calculate performance metrics
        total_return = ((capital - initial_capital) / initial_capital) * 100
        
        # Equity curve untuk plotting
        equity_index = df_bt.index[1:len(equity_curve)+1]
        equity_series = pd.Series(equity_curve, index=equity_index)
        
        # Calculate drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Win rate
        if trades:
            winning_trades = [t for t in trades if t['pnl_pct'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            # Profit factor
            total_profit = sum([t['pnl_pct'] for t in trades if t['pnl_pct'] > 0])
            total_loss = abs(sum([t['pnl_pct'] for t in trades if t['pnl_pct'] < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        # Create equity curve plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode='lines',
            name='Equity',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            height=400,
            showlegend=True
        )
        
        # Prepare performance dictionary
        perf = {
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'win_rate': round(win_rate, 2),
            'total_trades': len(trades),
            'profit_factor': round(profit_factor, 2),
            'trades_df': pd.DataFrame(trades) if trades else None,
            'equity_curve': equity_series
        }
        
        return perf, fig
        
    except Exception as e:
        print(f"Error in run_simple_backtest: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty results
        fig = go.Figure()
        fig.update_layout(title='Backtest Error', height=400)
        
        return {
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'total_trades': 0,
            'profit_factor': 0,
            'trades_df': None,
            'equity_curve': pd.Series()
        }, fig

# -------------------------
# NORMALIZE OHLCV SAFELY (FIXED)
# -------------------------
def normalize_ohlcv(df):
    df = df.copy()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c in df.columns:
            try:
                val = df[c]
                # Jika DataFrame, ambil kolom pertama
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                # Jika ndarray, ravel jadi 1D
                elif isinstance(val, np.ndarray):
                    val = np.ravel(val)
                    val = pd.Series(val[:len(df)], index=df.index[:len(val)])
                # Jika bukan Series, buat Series
                elif not isinstance(val, pd.Series):
                    val = pd.Series(val, index=df.index[:len(val)])
                # paksa numeric
                df[c] = pd.to_numeric(val, errors="coerce")
            except Exception:
                df[c] = np.nan
        else:
            df[c] = np.nan
    return df

# -------------------------
# SAFE CANDLE PLOT
# -------------------------
def plot_candlestick_safe(fig, df, name='OHLC'):
    df = clean_duplicate_columns(df.copy())
    
    def get_col_safe(col_name):
        if col_name in df.columns:
            try:
                col_data = df[col_name]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                elif not isinstance(col_data, pd.Series):
                    col_data = pd.Series(col_data, index=df.index[:len(df)])
                return np.asarray(col_data).reshape(-1)
            except Exception:
                return np.full(len(df), np.nan)
        else:
            return np.full(len(df), np.nan)
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=get_col_safe('open'),
        high=get_col_safe('high'),
        low=get_col_safe('low'),
        close=get_col_safe('close'),
        name=name
    ))
    return fig

# --------------------------------------------------
# MODEL LOADING (sebelum mode selection)
# --------------------------------------------------
MODEL_PATH = "models/best_lstm_model.h5"
SCALER_PATH = "models/minmax_scaler.pkl"
METADATA_PATH = "models/model_metadata.json"

# Initialize model variables
model = None
scaler = None
meta = None
model_loaded = False

# Try load model & scaler sekali di awal
try:
    model, scaler, meta = load_model_and_scaler(MODEL_PATH, SCALER_PATH, METADATA_PATH)
    model_loaded = True
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model/Scaler not loaded: {e}")
    model_loaded = False

# Di bagian model loading, setelah load metadata:
if meta and "features_used" not in meta:
    # Coba rename jika menggunakan kunci yang berbeda
    if "features" in meta:
        meta["features_used"] = meta["features"]
        print("Renamed 'features' to 'features_used' in metadata")


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Price Analysis PRO",
    page_icon="📈"
)

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------
st.markdown("""
<h1 style="margin-bottom: -20px;">📈 Price Analysis — <span style="color:#ff4b4b">PRO Edition</span></h1>
<p style="color:gray;">Full-featured technical analysis suite with prediction, indicators, divergences, regimes, S/R, and backtesting.</p>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONFIG
# --------------------------------------------------
# Di dalam with st.sidebar: (pastikan indentasi benar)
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    symbol = st.text_input("Symbol", value="BTC-USD")
    days = st.slider("History (days)", 30, 2000, 365)
    mode = st.radio("Mode", ["Basic", "PRO"], index=0)    
    
    st.markdown("---")
    st.markdown("### 🎯 Screener Thresholds")
    rsi_buy = st.number_input("RSI Buy", 5, 95, 30)
    rsi_sell = st.number_input("RSI Sell", 5, 95, 70)
    
    st.markdown("---")
    st.markdown("### 🧠 Model Status")
    if model_loaded:
        st.success("✅ LSTM Model Loaded")
        # Tampilkan info model
        with st.expander("Model Info"):
            if meta:
                st.write(f"Features: {len(meta.get('features_used', []))}")
                st.write(f"Window Size: {meta.get('best_parameters', {}).get('window_size', 'N/A')}")
    else:
        st.warning("⚠️ LSTM Model Not Loaded")

# --------------------------------------------------
# LOAD DATA - FIXED VERSION
# --------------------------------------------------
st.info(f"Fetching data from **Yahoo Finance** for: **{symbol}**", icon="📡")

try:
    # Download data dari Yahoo Finance
    df = yf.download(symbol, period=f"{days}d", interval="1d", auto_adjust=False)
    
    print(f"\n{'='*60}")
    print("RAW DATA FROM YAHOO FINANCE")
    print(f"{'='*60}")
    print(f"Type of columns: {type(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # CEK: Apakah ini MultiIndex?
    print(f"Is MultiIndex: {isinstance(df.columns, pd.MultiIndex)}")
    
    # HANDLE MULTIINDEX DENGAN BENAR
    if isinstance(df.columns, pd.MultiIndex):
        print("Processing MultiIndex columns...")
        
        # Cara 1: Flatten dengan mengambil level pertama saja
        # df.columns = df.columns.get_level_values(0)
        
        # Cara 2: Flatten dengan menggabungkan kedua level
        df.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df.columns]
        
        print(f"After flattening: {df.columns.tolist()}")
    
    # Hapus duplikat kolom
    df = clean_duplicate_columns(df)
    print(f"After removing duplicates: {df.columns.tolist()}")
    
    # Debug: Tampilkan info lengkap
    print(f"\nDataFrame Info:")
    print(f"Index: {df.index[:5] if len(df) > 0 else 'Empty'}")
    print(f"Data types:")
    for col in df.columns:
        print(f"  {col}: {type(df[col])}")
    
    # STANDARDIZE COLUMN NAMES - DENGAN CARA YANG LEBIH AMAN
    # Buat DataFrame baru untuk menampung kolom yang sudah distandardisasi
    new_df = pd.DataFrame(index=df.index)
    
    # Mapping kolom berdasarkan pola
    column_patterns = {
        'open': ['open', 'Open', 'OPEN'],
        'high': ['high', 'High', 'HIGH'], 
        'low': ['low', 'Low', 'LOW'],
        'close': ['close', 'Close', 'CLOSE'],
        'volume': ['volume', 'Volume', 'VOLUME']
    }
    
    for std_name, patterns in column_patterns.items():
        for pattern in patterns:
            # Cari kolom yang mengandung pattern
            matching_cols = [col for col in df.columns if pattern in str(col)]
            if matching_cols:
                # Ambil kolom pertama yang match
                source_col = matching_cols[0]
                print(f"Using {source_col} for {std_name}")
                
                # Salin data dengan konversi yang aman
                if source_col in df.columns:
                    # Ambil data sebagai Series
                    col_data = df[source_col]
                    
                    # Jika ini DataFrame, ambil kolom pertama
                    if isinstance(col_data, pd.DataFrame):
                        col_data = col_data.iloc[:, 0]
                    
                    # Konversi ke Series jika belum
                    if not isinstance(col_data, pd.Series):
                        col_data = pd.Series(col_data, index=df.index)
                    
                    # Konversi ke numeric
                    new_df[std_name] = pd.to_numeric(col_data, errors='coerce')
                break
        else:
            # Jika tidak ditemukan, buat kolom kosong
            print(f"Column for {std_name} not found, creating empty")
            new_df[std_name] = np.nan
    
    # Ganti df dengan new_df yang sudah distandardisasi
    df = new_df.copy()
    
    print(f"\n{'='*60}")
    print("AFTER STANDARDIZATION")
    print(f"{'='*60}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Isi NaN
    df = df.ffill().bfill()
    
    # Tampilkan sample data
    if len(df) > 0:
        print(f"\nSample data (first 3 rows):")
        print(df.head(3))
    
    # TAMBAH INDIKATOR DENGAN CARA YANG LEBIH AMAN
    print(f"\n{'='*60}")
    print("ADDING INDICATORS")
    print(f"{'='*60}")
    
    # Buat DataFrame sementara dengan kolom kapital untuk add_indicators
    df_for_indicators = df.copy()
    
    # Pastikan kolom dengan nama kapital ada
    kapital_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low', 
        'close': 'Close',
        'volume': 'Volume'
    }
    
    for lower, kapital in kapital_map.items():
        if lower in df_for_indicators.columns and kapital not in df_for_indicators.columns:
            df_for_indicators[kapital] = df_for_indicators[lower]
    
    print(f"Columns before add_indicators: {df_for_indicators.columns.tolist()}")
    
    # Coba panggil add_indicators
    try:
        df_with_indicators = add_indicators(df_for_indicators)
        
        # Gabungkan dengan df asli
        for col in df_with_indicators.columns:
            if col not in df.columns:
                df[col] = df_with_indicators[col]
        
        print(f"✅ Indicators added successfully")
        print(f"Total columns after indicators: {len(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error in add_indicators: {e}")
        # Tambahkan kolom indikator default secara manual
        for col in ['RSI_14', 'MACD', 'SMA_10', 'EMA_12', 'BB_low', 'BB_high']:
            if col not in df.columns:
                df[col] = np.nan
    
    # Pastikan kolom penting ada
    ensure_columns(df, ['close', 'RSI_14', 'MACD', 'BB_low', 'BB_high'])
    
    print(f"\n{'='*60}")
    print("FINAL DATAFRAME")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    
    if 'close' in df.columns:
        print(f"close column - dtype: {df['close'].dtype}, non-null: {df['close'].notna().sum()}")
    
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Buat DataFrame minimal sebagai fallback
    date_range = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    df = pd.DataFrame(index=date_range)
    for col in ['open', 'high', 'low', 'close', 'volume', 'RSI_14', 'MACD', 'BB_low', 'BB_high']:
        df[col] = np.nan

# --------------------------------------------------
# LATEST FORECAST FROM INFLUX
# --------------------------------------------------
pred = get_latest_forecast()
pred_value = None
if pred:
    for key in ["pred_close", "predicted", "forecast"]:
        if key in pred:
            try:
                pred_value = float(pred[key])
                break
            except Exception:
                pass

# --------------------------------------------------
# PREDIKSI LSTM (DIGUNAKAN OLEH BOTH MODES)
# --------------------------------------------------
lstm_predictions_series = None
lstm_next_prediction = None

if model_loaded and 'close' in df.columns and len(df) > 50:
    try:        
        # Gunakan fungsi prediksi yang sama untuk kedua mode
        lstm_predictions_series = predict_with_influx_style(df.copy(), model, scaler, meta)
        
        if lstm_predictions_series is not None and not lstm_predictions_series.empty:
            lstm_next_prediction = lstm_predictions_series.iloc[-1]
        else:
            st.warning("⚠️ LSTM predictions returned empty")
            
    except Exception as e:
        st.error(f"❌ Error calculating LSTM predictions: {e}")
        import traceback
        traceback.print_exc()
else:
    st.info("ℹ️ LSTM predictions disabled - model not loaded or insufficient data")

# --------------------------------------------------
# BASIC MODE
# --------------------------------------------------
if mode == "Basic":
    st.markdown("## 📊 Basic Overview - LSTM Synchronized")
    
    # Price chart
    fig = go.Figure()
    fig = plot_candlestick_safe(fig, df)
    
    # --- PREDIKSI YANG SAMA DENGAN PRO MODE ---
    basic_pred_value = None
    pred_source = "N/A"
    
    # PRIORITAS 1: Prediksi LSTM (sama dengan Pro mode)
    if lstm_next_prediction is not None:
        basic_pred_value = lstm_next_prediction
        pred_source = "LSTM Model"
        
        # Tambahkan garis prediksi di chart
        fig.add_hline(
            y=basic_pred_value,
            line_dash="dot",
            line_width=2,
            annotation_text=f"LSTM Forecast: {basic_pred_value:.2f}",
            annotation_position="bottom right",
            line_color="orange",
            annotation_font_color="orange"
        )
        
        # Jika mau, tambahkan juga full prediction series (opsional)
        if lstm_predictions_series is not None and len(lstm_predictions_series) > 0:
            fig.add_trace(go.Scatter(
                x=lstm_predictions_series.index,
                y=lstm_predictions_series.values,
                mode='lines',
                name='LSTM Trend',
                line=dict(color='orange', width=1, dash='dash')
            ))
    
    # PRIORITAS 2: Influx (jika LSTM tidak tersedia)
    elif pred_value is not None:
        basic_pred_value = pred_value
        pred_source = "Influx"
        
        fig.add_hline(
            y=basic_pred_value,
            line_dash="dot",
            line_width=2,
            annotation_text=f"Influx Forecast: {basic_pred_value:.2f}",
            annotation_position="top right",
            line_color="green",
            annotation_font_color="green"
        )
    
    fig.update_layout(
        height=600, 
        margin=dict(t=10, b=10),
        title=f"{symbol} Price with {pred_source} Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats yang diperkaya
    st.markdown("### 📌 Market Stats & Predictions")
    col1, col2, col3, col4 = st.columns(4)
    
    last_close = safe_last_float(df['close'], default=np.nan)
    col1.metric("Latest Close", f"${last_close:,.2f}" if not np.isnan(last_close) else "N/A")
    
    # Kolom prediksi
    with col2:
        if basic_pred_value is not None:
            st.metric(f"Next ({pred_source})", f"${basic_pred_value:,.2f}")
        else:
            st.metric("Next Prediction", "N/A")
    
    # Kolom perubahan
    with col3:
        if basic_pred_value is not None and not np.isnan(last_close) and last_close > 0:
            change_pct = ((basic_pred_value - last_close) / last_close) * 100
            st.metric(
                "Expected Change", 
                f"{change_pct:+.2f}%",
                delta=f"{change_pct:+.2f}%"
            )
        else:
            st.metric("Expected Change", "N/A")
    
    # Kolom RSI
    last_rsi = safe_last_float(df.get('RSI_14', pd.Series(dtype=float)))
    with col4:
        if not np.isnan(last_rsi):
            # Tampilkan dengan warna berdasarkan level
            rsi_color = "green" if last_rsi < 30 else "red" if last_rsi > 70 else "gray"
            st.markdown(f"<p style='color:{rsi_color}; font-weight:bold;'>RSI (14): {last_rsi:.2f}</p>", 
                       unsafe_allow_html=True)
        else:
            st.metric("RSI (14)", "N/A")
        
    # Screener signals
    st.markdown("### 🔍 Screener Signals")
    df_screen = safe_call_screener(df, rsi_buy, rsi_sell)
    
    # Hapus duplikat kolom sebelum menampilkan
    df_screen = clean_duplicate_columns(df_screen)
    
    # Tampilkan dengan styling
    show_cols = []
    for col in ['close', 'RSI_14', 'MACD', 'signal_buy', 'signal_sell']:
        if col in df_screen.columns:
            show_cols.append(col)
    
    if show_cols:
        # Format tampilan
        display_df = df_screen[show_cols].tail(20).copy()
        
        # Style untuk signals
        def style_signals(val):
            if val.name == 'signal_buy' and val.any():
                return ['background-color: lightgreen' if v else '' for v in val]
            elif val.name == 'signal_sell' and val.any():
                return ['background-color: lightcoral' if v else '' for v in val]
            return [''] * len(val)
        
        st.dataframe(display_df.style.apply(style_signals, subset=['signal_buy', 'signal_sell']))
    else:
        st.warning("No screener data available")
    
    # Info prediksi LSTM
    if lstm_predictions_series is not None:
        with st.expander("📊 LSTM Prediction Details"):
            st.write(f"Prediction range: {len(lstm_predictions_series)} points")
            st.write(f"From: {lstm_predictions_series.index[0]} to {lstm_predictions_series.index[-1]}")
            st.line_chart(lstm_predictions_series.tail(50))

# --------------------------------------------------
# PRO MODE
# --------------------------------------------------
else:
    st.markdown("## 🟥 PRO Mode — Full Technical Suite")

    # Model sudah diload di bagian awal, jadi kita hanya perlu cek status
    if not model_loaded:
        st.warning("⚠️ LSTM Model not loaded. Some features may be limited.")
        st.info("Model path tried:")
        st.code(f"Model: {MODEL_PATH}\nScaler: {SCALER_PATH}\nMetadata: {METADATA_PATH}")

    # Ganti deklarasi tabs menjadi:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Price & Prediction",
        "🔻 Divergences", 
        "📊 Volume Profile",
        "📉 Regime + S/R + Backtest",
        "🔮 Future Forecast"
    ])

    # -------------------------
    # TAB 1 — Price & Prediction
    # -------------------------
    with tab1:
        st.markdown("### 📈 Price Chart & LSTM Predictions (Synced with Basic Mode)")
        
        # Pastikan DataFrame bersih
        df_clean = clean_duplicate_columns(df.copy())
        
        # Build candlestick chart
        fig = go.Figure()
        fig = plot_candlestick_safe(fig, df_clean)
        
        # Gunakan prediksi LSTM yang SAMA dengan Basic mode
        if lstm_predictions_series is not None and not lstm_predictions_series.empty:
            # Add full prediction series
            fig.add_trace(go.Scatter(
                x=lstm_predictions_series.index, 
                y=lstm_predictions_series.values,
                mode='lines', 
                name='LSTM Predictions',
                line=dict(color='orange', width=2)
            ))
            
            # Highlight the next prediction
            if lstm_next_prediction is not None:
                fig.add_trace(go.Scatter(
                    x=[lstm_predictions_series.index[-1]], 
                    y=[lstm_next_prediction],
                    mode='markers+text',
                    name='Next Prediction',
                    marker=dict(size=12, color='red'),
                    text=[f"Next: ${lstm_next_prediction:.2f}"],
                    textposition="top center"
                ))
            
            # Jika ada prediksi dari Influx, tampilkan perbandingan
            if pred_value is not None:
                fig.add_hline(
                    y=pred_value,
                    line_dash="dot",
                    line_width=1.5,
                    annotation_text=f"Influx: {pred_value:.2f}",
                    annotation_position="top right",
                    line_color="green"
                )
                
                # Tambahkan perbedaan visual
                diff = lstm_next_prediction - pred_value if lstm_next_prediction else 0
                fig.add_annotation(
                    x=lstm_predictions_series.index[-1],
                    y=(lstm_next_prediction + pred_value) / 2 if lstm_next_prediction else pred_value,
                    text=f"Δ: ${diff:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="blue"
                )
        
        fig.update_layout(
            height=620, 
            margin=dict(t=20, b=10),
            title=f"{symbol} - Synced LSTM Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics section - SAMA dengan Basic mode
        st.markdown("### 📊 Synchronized Prediction Metrics")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        last_close = df_clean['close'].iloc[-1] if 'close' in df_clean.columns else np.nan
        
        with metrics_col1:
            st.metric("Last Close", f"${last_close:,.2f}" if not np.isnan(last_close) else "N/A")
        
        with metrics_col2:
            if lstm_next_prediction is not None:
                st.metric("LSTM Next", f"${lstm_next_prediction:,.2f}")
            else:
                st.metric("LSTM Next", "N/A")
        
        with metrics_col3:
            if lstm_next_prediction is not None and not np.isnan(last_close) and last_close > 0:
                change = ((lstm_next_prediction - last_close) / last_close * 100)
                st.metric(
                    "LSTM Change", 
                    f"{change:.2f}%",
                    delta=f"{change:.2f}%",
                    delta_color="normal" if change > 0 else "inverse"
                )
            else:
                st.metric("Change", "N/A")
        
        # Tabel prediksi terbaru
        if lstm_predictions_series is not None:
            st.markdown("#### 📋 Recent Predictions (Same as Basic Mode)")
            
            # Buat DataFrame dengan prediksi dan actual
            pred_df = pd.DataFrame({
                'Date': lstm_predictions_series.index[-20:],
                'LSTM Prediction': lstm_predictions_series.values[-20:].round(2)
            })
            
            # Tambahkan actual close jika tersedia
            if 'close' in df_clean.columns:
                # Align dates
                actual_prices = []
                for date in pred_df['Date']:
                    if date in df_clean.index:
                        actual_prices.append(df_clean.loc[date, 'close'])
                    else:
                        # Find nearest date
                        if date < df_clean.index[0]:
                            actual_prices.append(df_clean['close'].iloc[0])
                        elif date > df_clean.index[-1]:
                            actual_prices.append(df_clean['close'].iloc[-1])
                        else:
                            # Find closest date
                            idx = df_clean.index.get_indexer([date], method='nearest')[0]
                            actual_prices.append(df_clean['close'].iloc[idx])
                
                pred_df['Actual Close'] = actual_prices
                pred_df['Difference'] = pred_df['LSTM Prediction'] - pred_df['Actual Close']
                pred_df['Diff %'] = (pred_df['Difference'] / pred_df['Actual Close'] * 100).round(2)
            
            st.dataframe(pred_df, hide_index=True)
            
            # Grafik akurasi prediksi
            if 'Actual Close' in pred_df.columns:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    x=pred_df['Date'], y=pred_df['Actual Close'],
                    mode='lines+markers', name='Actual',
                    line=dict(color='blue', width=2)
                ))
                fig_acc.add_trace(go.Scatter(
                    x=pred_df['Date'], y=pred_df['LSTM Prediction'],
                    mode='lines+markers', name='LSTM Pred',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                fig_acc.update_layout(
                    height=400,
                    title="Prediction vs Actual (Last 20 points)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    showlegend=True
                )
                st.plotly_chart(fig_acc, use_container_width=True)
        
    # -------------------------
    # TAB 2 — Divergences
    # -------------------------
    with tab2:
        st.markdown("### 🔻 Divergence Detector")
        indicator = st.selectbox("Indicator", ["RSI_14", "MACD"], index=0)
        look = st.slider("Lookback (bars)", 20, 150, 60)

        # Pastikan indikator yang dipilih ada di DataFrame
        if indicator not in df.columns:
            available_indicators = [col for col in df.columns if 'RSI' in col or 'MACD' in col]
            st.error(f"Indicator '{indicator}' not found in data. Available indicators: {available_indicators}")
        else:
            # Clean df before divergence detection
            df_clean_div = clean_duplicate_columns(df.copy())
            divdf = safe_detect_divergences(df_clean_div, lookback=look, indicator=indicator)
            
            if divdf is None or divdf.empty:
                st.info("No divergence data available (not enough data or indicator missing).")
            else:
                # Clean the results
                divdf = clean_duplicate_columns(divdf)
                
                figd = go.Figure()
                
                # Plot candlestick
                try:
                    if all(col in divdf.columns for col in ['open', 'high', 'low', 'close']):
                        figd = plot_candlestick_safe(figd, divdf)
                    else:
                        # Fallback: plot close line
                        if 'close' in divdf.columns:
                            figd.add_trace(go.Scatter(
                                x=divdf.index,
                                y=divdf['close'],
                                mode='lines',
                                name='Close',
                                line=dict(color='blue', width=1)
                            ))
                        else:
                            st.error("No OHLC data available for plotting")
                except Exception as e:
                    st.error(f"Error plotting candlestick: {e}")
                
                # Plot divergences if columns exist
                if 'bull_divergence' in divdf.columns:
                    bulls = divdf[divdf['bull_divergence']]
                    bears = divdf[divdf['bear_divergence']]
                    
                    if not bulls.empty and 'low' in divdf.columns:
                        try:
                            low_vals = np.asarray(bulls['low'], dtype=float).reshape(-1)
                            figd.add_trace(go.Scatter(
                                x=bulls.index, 
                                y=low_vals,
                                mode='markers', 
                                marker=dict(size=12, color='green', symbol='triangle-up'),
                                name='Bull Div'
                            ))
                        except Exception as e:
                            st.warning(f"Could not plot bull divergences: {e}")
                    
                    if not bears.empty and 'high' in divdf.columns:
                        try:
                            high_vals = np.asarray(bears['high'], dtype=float).reshape(-1)
                            figd.add_trace(go.Scatter(
                                x=bears.index, 
                                y=high_vals,
                                mode='markers', 
                                marker=dict(size=12, color='red', symbol='triangle-down'),
                                name='Bear Div'
                            ))
                        except Exception as e:
                            st.warning(f"Could not plot bear divergences: {e}")
                
                figd.update_layout(
                    height=600,
                    title=f"Divergence Detection ({indicator}, lookback={look})",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    showlegend=True
                )
                st.plotly_chart(figd, use_container_width=True)
                
                # Show divergence statistics
                if 'bull_divergence' in divdf.columns and 'bear_divergence' in divdf.columns:
                    bull_count = divdf['bull_divergence'].sum()
                    bear_count = divdf['bear_divergence'].sum()
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Bullish Divergences", int(bull_count))
                    col2.metric("Bearish Divergences", int(bear_count))
    
    # -------------------------
    # TAB 3 — Volume Profile
    # -------------------------
    with tab3:
        st.markdown("### 📊 Volume Profile")
        bins = st.slider("Bins", 10, 80, 30)
        try:
            vp = compute_volume_profile(df, bins=bins)
            figvp = px.bar(vp.sort_values('price_left'), x='volume', y='price_left', orientation='h',
                           labels={'volume': 'Volume', 'price_left': 'Price'}, title="Volume Profile")
            figvp.update_layout(height=600)
            st.plotly_chart(figvp, use_container_width=True)
        except Exception as e:
            st.error(f"Volume profile failed: {e}")
    
    # -------------------------
    # TAB 4 — Regime, S/R, Backtest
    # -------------------------
    with tab4:
        st.markdown("### 📉 Market Regime + Support/Resistance + Backtest LSTM")
        
        # Pastikan DataFrame bersih
        df_clean = clean_duplicate_columns(df.copy())
        
        # Market Regime
        st.markdown("#### 📈 Market Regime")
        try:
            reg = detect_market_regime(df_clean.copy())
            if isinstance(reg, pd.DataFrame) and 'regime' in reg.columns:
                latest = str(reg['regime'].iloc[-1])
                st.success(f"Current regime (last bar): **{latest.upper()}**")
                
                # Show regime timeline
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(x=reg.index, y=reg['close'], mode='lines', name='Close'))
                fig_r.update_layout(height=260, title="Market Regime")
                st.plotly_chart(fig_r, use_container_width=True)
            else:
                st.info("Market regime not available.")
        except Exception as e:
            st.info(f"Market regime detection failed: {e}")
        
        # Support & Resistance
        st.markdown("#### 🏗️ Support & Resistance")
        sup_list, res_list = safe_find_support_resistance(df_clean, n=6)
        
        col_sup, col_res = st.columns(2)
        with col_sup:
            st.subheader("Supports (recent)")
            if sup_list:
                df_sup = pd.DataFrame(sup_list, columns=['time', 'price']).set_index('time').sort_index(ascending=False)
                st.dataframe(df_sup)
            else:
                st.info("No support levels found.")
        
        with col_res:
            st.subheader("Resistances (recent)")
            if res_list:
                df_res = pd.DataFrame(res_list, columns=['time', 'price']).set_index('time').sort_index(ascending=False)
                st.dataframe(df_res)
            else:
                st.info("No resistance levels found.")
        
        # Backtest LSTM
        st.markdown("#### 📊 Backtest LSTM")
        
        if not model_loaded:
            st.warning("Model not loaded — cannot run LSTM backtest.")
        else:
            # Signal Mode selection
            st.markdown("##### ⚙️ Signal Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                signal_mode = st.selectbox("Signal Mode", ["momentum", "directional", "crossover"], index=0,
                                        help="momentum: pred[t] > pred[t-1] | directional: pred[t] > close[t-1] | crossover: pred cross sma(pred)")
            with col2:
                use_hybrid = st.checkbox("Use Hybrid: RSI + LSTM", value=True,
                                    help="Combine LSTM signals with RSI overbought/oversold conditions")
            with col3:
                use_regime_filter = st.checkbox("Only trade in Bull regime", value=False,
                                            help="Only take trades when market regime is bullish")
            
            if use_hybrid:
                col4, col5 = st.columns(2)
                with col4:
                    rsi_buy_thr = st.number_input("RSI buy threshold", value=30, min_value=1, max_value=49,
                                                help="RSI level for buy signal (oversold)")
                with col5:
                    rsi_sell_thr = st.number_input("RSI sell threshold", value=70, min_value=51, max_value=99,
                                                help="RSI level for sell signal (overbought)")
            
            # Tambahkan parameter tambahan
            st.markdown("##### 🎛️ Backtest Parameters")
            col_param1, col_param2, col_param3 = st.columns(3)
            
            with col_param1:
                initial_capital = st.number_input("Initial Capital ($)", value=1000.0, min_value=100.0, step=100.0)
            
            with col_param2:
                transaction_fee = st.number_input("Transaction Fee (%)", value=0.1, min_value=0.0, max_value=1.0, step=0.01) / 100
            
            with col_param3:
                stop_loss_pct = st.number_input("Stop Loss (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5) / 100
                take_profit_pct = st.number_input("Take Profit (%)", value=10.0, min_value=1.0, max_value=50.0, step=0.5) / 100
            
            if st.button("🔄 Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    try:
                        # Build features for backtest
                        df_clean_bt = clean_duplicate_columns(df.copy())
                        
                        # Pastikan kolom 'close' ada
                        if 'close' not in df_clean_bt.columns and 'Close' in df_clean_bt.columns:
                            df_clean_bt['close'] = df_clean_bt['Close']
                        
                        # Debug: tampilkan info data
                        st.info(f"Data for backtest: {len(df_clean_bt)} bars, from {df_clean_bt.index[0]} to {df_clean_bt.index[-1]}")
                        
                        # Get features for model
                        df_feat = build_features_for_model(df_clean_bt.copy(), meta)
                        
                        if df_feat.empty:
                            st.error("Feature DataFrame is empty. Cannot make predictions.")
                        else:
                            # Make predictions
                            st.info("Making predictions...")
                            preds = simple_predict_fix(df_feat.copy(), model, scaler, meta)
                            
                            if preds.empty:
                                st.error("No predictions generated. Check model and data.")
                                st.stop()
                            
                            # Align predictions with original dataframe
                            preds = preds.reindex(df_clean_bt.index)
                            df_clean_bt['prediction'] = preds
                            
                            # Debug: tampilkan beberapa prediksi
                            with st.expander("🔍 Prediction Preview"):
                                st.write(f"Predictions generated: {len(preds.dropna())}")
                                st.dataframe(df_clean_bt[['close', 'prediction']].tail(20))
                            
                            # Build signals according to chosen mode
                            df_bt = df_clean_bt.copy()
                            
                            # Momentum signal
                            if signal_mode == "momentum":
                                df_bt["signal"] = 0
                                # Buy when prediction is increasing
                                df_bt.loc[df_bt["prediction"] > df_bt["prediction"].shift(1), "signal"] = 1
                                # Sell when prediction is decreasing
                                df_bt.loc[df_bt["prediction"] < df_bt["prediction"].shift(1), "signal"] = -1
                            
                            # Directional signal
                            elif signal_mode == "directional":
                                df_bt["prediction"] = df_bt["prediction"].astype(float)
                                df_bt["signal"] = 0
                                # Buy when prediction > previous close
                                df_bt.loc[df_bt["prediction"] > df_bt["close"].shift(1), "signal"] = 1
                                # Sell when prediction < previous close
                                df_bt.loc[df_bt["prediction"] < df_bt["close"].shift(1), "signal"] = -1
                            
                            # Crossover signal
                            else:  # crossover
                                window_c = int(meta.get("window", 14)) if meta else 14
                                df_bt["pred_sma"] = df_bt["prediction"].rolling(window=window_c, min_periods=1).mean()
                                df_bt["signal"] = 0
                                # Golden cross: prediction crosses above SMA
                                cross_up = (df_bt["prediction"] > df_bt["pred_sma"]) & (df_bt["prediction"].shift(1) <= df_bt["pred_sma"].shift(1))
                                # Death cross: prediction crosses below SMA
                                cross_dn = (df_bt["prediction"] < df_bt["pred_sma"]) & (df_bt["prediction"].shift(1) >= df_bt["pred_sma"].shift(1))
                                df_bt.loc[cross_up, "signal"] = 1
                                df_bt.loc[cross_dn, "signal"] = -1
                            
                            # Debug: tampilkan sinyal
                            with st.expander("🔍 Signal Preview"):
                                st.write(f"Signal distribution:")
                                st.write(df_bt['signal'].value_counts())
                                st.dataframe(df_bt[['close', 'prediction', 'signal']].tail(20))
                            
                            # Hybrid filter (RSI + LSTM)
                            if use_hybrid:
                                # Ensure RSI exists
                                if 'RSI_14' not in df_bt.columns:
                                    df_bt = safe_add_indicators(df_bt)
                                    df_bt = clean_duplicate_columns(df_bt)
                                
                                if 'RSI_14' in df_bt.columns:
                                    df_bt['RSI_14'] = df_bt['RSI_14'].astype(float)
                                    
                                    # Original signals
                                    original_signals = df_bt['signal'].copy()
                                    
                                    # Apply RSI filters
                                    # Buy: LSTM says buy AND RSI is oversold
                                    buy_condition = (original_signals == 1) & (df_bt['RSI_14'] <= rsi_buy_thr)
                                    # Sell: LSTM says sell AND RSI is overbought
                                    sell_condition = (original_signals == -1) & (df_bt['RSI_14'] >= rsi_sell_thr)
                                    
                                    # Reset all signals first
                                    df_bt['signal'] = 0
                                    
                                    # Apply filtered signals
                                    df_bt.loc[buy_condition, 'signal'] = 1
                                    df_bt.loc[sell_condition, 'signal'] = -1
                                    
                                    st.info(f"Hybrid filter applied: {buy_condition.sum()} buy signals, {sell_condition.sum()} sell signals")
                                else:
                                    st.warning("RSI column not found, skipping hybrid filter")
                            
                            # Regime filter
                            if use_regime_filter and isinstance(reg, pd.DataFrame) and 'regime' in reg.columns:
                                reg_aligned = reg.reindex(df_bt.index).fillna(method='ffill').fillna(0)
                                # Assume 1 = Bull regime
                                non_bull_mask = reg_aligned['regime'] != 1
                                df_bt.loc[non_bull_mask, 'signal'] = 0
                                st.info(f"Regime filter applied: {non_bull_mask.sum()} bars filtered out")
                            
                            # Clean duplicates before backtest
                            df_bt = clean_duplicate_columns(df_bt)
                            
                            # Run backtest dengan fungsi yang diperbaiki
                            st.info("Running backtest...")
                            
                            # Gunakan fungsi backtest yang lebih sederhana dan reliable
                            perf, fig_equity = run_simple_backtest(
                                df_bt, 
                                initial_capital=initial_capital, 
                                fee=transaction_fee,
                                stop_loss_pct=stop_loss_pct,
                                take_profit_pct=take_profit_pct
                            )
                            
                            # Display results
                            st.markdown("##### 📈 Equity Curve")
                            st.plotly_chart(fig_equity, use_container_width=True)
                            
                            st.markdown("##### 📊 Performance Metrics")
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            
                            with metrics_col1:
                                st.metric("Total Return", f"{perf.get('total_return', 0):.2f}%")
                                st.metric("Sharpe Ratio", f"{perf.get('sharpe_ratio', 0):.2f}")
                            
                            with metrics_col2:
                                st.metric("Max Drawdown", f"{perf.get('max_drawdown', 0):.2f}%")
                                st.metric("Win Rate", f"{perf.get('win_rate', 0):.2f}%")
                            
                            with metrics_col3:
                                st.metric("Total Trades", f"{perf.get('total_trades', 0)}")
                                st.metric("Profit Factor", f"{perf.get('profit_factor', 0):.2f}")
                            
                            # Trades table
                            if perf.get('trades_df') is not None and not perf['trades_df'].empty:
                                st.markdown("##### 📝 Recent Trades")
                                trades_df = perf['trades_df'].sort_values('entry_time', ascending=False).head(10)
                                st.dataframe(trades_df)
                                
                                # Trade statistics
                                st.markdown("##### 📈 Trade Statistics")
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                
                                with col_stat1:
                                    avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl_pct'] > 0]) > 0 else 0
                                    avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl_pct'] < 0]) > 0 else 0
                                    st.metric("Avg Win", f"{avg_win:.2f}%")
                                    st.metric("Avg Loss", f"{avg_loss:.2f}%")
                                
                                with col_stat2:
                                    best_trade = trades_df['pnl_pct'].max() if not trades_df.empty else 0
                                    worst_trade = trades_df['pnl_pct'].min() if not trades_df.empty else 0
                                    st.metric("Best Trade", f"{best_trade:.2f}%")
                                    st.metric("Worst Trade", f"{worst_trade:.2f}%")
                                
                                with col_stat3:
                                    total_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days if not trades_df.empty else 0
                                    trades_per_day = len(trades_df) / total_days if total_days > 0 else 0
                                    st.metric("Total Days", f"{total_days}")
                                    st.metric("Trades/Day", f"{trades_per_day:.2f}")
                            else:
                                st.info("No trades generated by the current settings.")
                                
                                # Debug: tampilkan sinyal untuk analisis
                                with st.expander("🔍 Debug: Signal Analysis"):
                                    st.write("Signal value counts:")
                                    st.write(df_bt['signal'].value_counts())
                                    
                                    st.write("Sample signals (last 50):")
                                    st.dataframe(df_bt[['close', 'prediction', 'signal']].tail(50))
                                    
                                    # Coba identifikasi masalah
                                    if (df_bt['signal'] == 0).all():
                                        st.error("All signals are 0. Check signal generation logic.")
                                    elif (df_bt['signal'].abs().sum() == 0):
                                        st.error("No non-zero signals. Check thresholds and conditions.")
                    
                    except Exception as e:
                        st.error(f"Backtest LSTM failed: {e}")
                        with st.expander("Debug Details"):
                            st.write(f"Error type: {type(e).__name__}")
                            st.write(f"Error message: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())

    # -------------------------
    # TAB 5 — Future Forecast (7-30 Days)
    # -------------------------
    with tab5:
        st.markdown("## 🔮 Future Price Forecast")
        
        # Info menggunakan model yang sama
        if model_loaded:
            st.success(f"✅ Menggunakan model LSTM yang sudah dilatih untuk forecasting")
            with st.expander("📋 Model Details"):
                if meta:
                    st.write(f"**Features:** {len(meta.get('features_used', []))}")
                    st.write(f"**Window Size:** {meta.get('best_parameters', {}).get('window_size', 'N/A')}")
                    st.write(f"**Model Type:** LSTM")
        else:
            st.error("❌ Model LSTM tidak tersedia. Forecasting tidak dapat dilakukan.")
            st.stop()
        
        # Configuration
        st.markdown("### ⚙️ Forecast Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_days = st.slider(
                "Forecast Period (Days)",
                min_value=7,
                max_value=30,
                value=7,
                step=1,
                help="Jumlah hari ke depan yang akan diprediksi"
            )
        
        with col2:
            use_confidence = st.checkbox(
                "Show Confidence Interval",
                value=True,
                help="Tampilkan interval kepercayaan untuk prediksi"
            )
        
        with col3:
            show_metrics = st.checkbox(
                "Show Detailed Metrics",
                value=True,
                help="Tampilkan metrik detail forecasting"
            )
        
        # Tombol untuk menjalankan forecasting
        if st.button("🚀 Run Future Forecast", type="primary"):
            with st.spinner(f"🔄 Forecasting {forecast_days} hari ke depan..."):
                try:
                    # Siapkan data untuk forecasting
                    df_clean = clean_duplicate_columns(df.copy())
                    
                    # **SOLUSI: HITUNG FORECASTING LANGSUNG DI SINI**
                    # 1. Pastikan ada data yang cukup
                    if len(df_clean) < 50:
                        st.warning(f"⚠️ Data historis hanya {len(df_clean)} hari. Minimal 50 hari untuk forecasting.")
                    
                    # 2. Pastikan kolom 'close' ada
                    if 'close' not in df_clean.columns:
                        if 'Close' in df_clean.columns:
                            df_clean['close'] = df_clean['Close']
                        else:
                            # Cari kolom yang mungkin berisi harga
                            for col in ['close', 'Close', 'price', 'Price', 'adj close', 'Adj Close', 'last']:
                                if col in df_clean.columns:
                                    df_clean['close'] = df_clean[col]
                                    break
                            else:
                                # Gunakan kolom numerik pertama
                                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                                if len(numeric_cols) > 0:
                                    df_clean['close'] = df_clean[numeric_cols[0]]
                                else:
                                    st.error("❌ Tidak ditemukan kolom harga untuk forecasting")
                                    st.stop()
                    
                    # 3. Bangun fitur menggunakan fungsi yang sudah ada di file ini
                    # (karena build_features_for_model sudah ada di file utama)                    
                    # Gunakan fungsi build_features_for_model yang sudah ada di file ini
                    df_features = build_features_for_model(df_clean.copy(), meta)
                    
                    if df_features.empty:
                        st.error("❌ Tidak dapat membangun fitur untuk forecasting")
                        st.stop()
                                    
                    # 4. Ambil parameter dari metadata
                    features_used = meta.get('features_used', [])
                    window_size = meta.get('best_parameters', {}).get('window_size', 10)
                                        
                    # 5. Scale data
                    scaled_data = scaler.transform(df_features)
                    
                    # 6. Ambil sequence terakhir
                    last_sequence = scaled_data[-window_size:]
                    
                    # 7. Cari index fitur 'close'
                    close_idx = 3  # default
                    for i, feat in enumerate(features_used):
                        if isinstance(feat, str) and 'close' in feat.lower():
                            close_idx = i
                            break
                                        
                    # 8. Buat forecasting secara rekursif
                    forecast_scaled = []
                    current_sequence = last_sequence.copy()
                    
                    for day in range(forecast_days):
                        input_seq = current_sequence.reshape(1, window_size, len(features_used))
                        pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
                        forecast_scaled.append(pred_scaled)
                        
                        # Update sequence
                        new_row = current_sequence[-1].copy()
                        new_row[close_idx] = pred_scaled
                        current_sequence = np.vstack([current_sequence[1:], new_row])
                    
                    # 9. Inverse transform prediksi
                    dummy_forecast = np.zeros((len(forecast_scaled), len(features_used)))
                    dummy_forecast[:, close_idx] = forecast_scaled
                    forecast_inversed = scaler.inverse_transform(dummy_forecast)
                    forecast_prices = forecast_inversed[:, close_idx]
                                        
                    # 10. **BUAT TANGGAL DENGAN CARA YANG BENAR**
                    last_date = df_clean.index[-1]
                    
                    # Pastikan last_date adalah Timestamp
                    if not isinstance(last_date, pd.Timestamp):
                        last_date = pd.to_datetime(last_date)
                    
                    # Buat tanggal forecast
                    forecast_dates = []
                    for i in range(1, forecast_days + 1):
                        # CARA YANG BENAR: Gunakan pd.DateOffset
                        next_date = last_date + pd.DateOffset(days=i)
                        forecast_dates.append(next_date)
                    
                    forecast_dates = pd.DatetimeIndex(forecast_dates)
                    
                    # Buat DataFrame hasil
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecast_Price': forecast_prices,
                        'Symbol': symbol
                    })
                    
                    st.success(f"✅ Forecast selesai! Prediksi {forecast_days} hari ke depan")
                    
                    # 11. Hitung metrik
                    current_price = float(df_clean['close'].iloc[-1])
                    
                    # Fungsi sederhana untuk menghitung metrik
                    def simple_calculate_metrics(prices, current):
                        metrics = {}
                        if len(prices) > 0:
                            # Total perubahan
                            if current > 0:
                                metrics['total_change_pct'] = ((prices[-1] - current) / current) * 100
                            else:
                                metrics['total_change_pct'] = 0
                            
                            # Min/Max
                            metrics['min_price'] = np.min(prices)
                            metrics['max_price'] = np.max(prices)
                            
                            # Trend
                            metrics['trend'] = 'UP' if metrics['total_change_pct'] > 0 else 'DOWN' if metrics['total_change_pct'] < 0 else 'FLAT'
                            
                            # Daily changes
                            if len(prices) > 1:
                                changes = []
                                for i in range(1, len(prices)):
                                    if prices[i-1] > 0:
                                        change = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
                                        changes.append(change)
                                
                                if changes:
                                    metrics['avg_daily_change'] = np.mean(changes)
                                    metrics['volatility'] = np.std(changes)
                                else:
                                    metrics['avg_daily_change'] = 0
                                    metrics['volatility'] = 0
                            else:
                                metrics['avg_daily_change'] = 0
                                metrics['volatility'] = 0
                        
                        return metrics
                    
                    metrics = simple_calculate_metrics(forecast_prices, current_price)
                    
                    # **LANJUTKAN DENGAN VISUALISASI (sama seperti sebelumnya)**
                    # 1. Visualisasi Forecast
                    st.markdown("### 📊 Forecast Visualization")
                    
                    fig = go.Figure()
                    
                    # Plot historical data (last 60 days)
                    hist_days = min(60, len(df_clean))
                    hist_dates = df_clean.index[-hist_days:]
                    hist_prices = df_clean['close'].values[-hist_days:]
                    
                    fig.add_trace(go.Scatter(
                        x=hist_dates,
                        y=hist_prices,
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue', width=2),
                        hovertemplate='%{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Plot forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Forecast_Price'],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2),
                        marker=dict(size=6),
                        hovertemplate='%{x|%Y-%m-%d}<br>Forecast: $%{y:.2f}<extra></extra>'
                    ))
                    
                    # Tambahkan garis vertikal pemisah
                    last_hist_date = hist_dates[-1]
                    fig.add_shape(
                        type="line",
                        x0=last_hist_date,
                        y0=0,
                        x1=last_hist_date,
                        y1=1,
                        yref="paper",  # referensi ke seluruh tinggi plot
                        line=dict(color="gray", dash="dash", width=2)
                    )
                    
                    fig.add_annotation(
                            x=last_hist_date,
                            y=1,
                            yref="paper",
                            text="Today",
                            showarrow=False,
                            xanchor="left",
                            yanchor="bottom",
                            font=dict(size=12, color="gray"),
                            bgcolor="white",
                            bordercolor="gray",
                            borderwidth=1,
                            borderpad=4,
                            opacity=0.8
                    )


                    fig.update_layout(
                        title=f'{symbol} {forecast_days}-Day Price Forecast',
                        xaxis_title='Date',
                        yaxis_title='Price (USD)',
                        height=500,
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Summary Metrics
                    st.markdown("### 📈 Forecast Summary")
                    
                    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                    
                    with col_sum1:
                        st.metric(
                            "Current Price",
                            f"${current_price:,.2f}",
                            delta=None
                        )
                    
                    with col_sum2:
                        forecast_end_price = forecast_prices[-1] if len(forecast_prices) > 0 else 0
                        st.metric(
                            f"Price in {forecast_days} days",
                            f"${forecast_end_price:,.2f}",
                            delta=f"{metrics.get('total_change_pct', 0):.2f}%"
                        )
                    
                    with col_sum3:
                        avg_change = metrics.get('avg_daily_change', 0)
                        st.metric(
                            "Avg Daily Change",
                            f"{avg_change:+.2f}%",
                            delta_color="normal"
                        )
                    
                    with col_sum4:
                        trend = metrics.get('trend', 'FLAT')
                        trend_color = "green" if trend == 'UP' else "red" if trend == 'DOWN' else "gray"
                        st.markdown(
                            f"<h3 style='color:{trend_color};'>Trend: {trend}</h3>",
                            unsafe_allow_html=True
                        )
                    
                    # 3. Detailed Forecast Table
                    st.markdown("### 📋 Detailed Forecast Table")
                    
                    # Buat tabel dengan informasi tambahan
                    display_df = forecast_df.copy()
                    
                    # Format tanggal
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                    display_df['Day'] = range(1, len(display_df) + 1)
                    
                    # Hitung perubahan harian
                    prices_numeric = forecast_prices
                    daily_changes = []
                    cumulative_changes = []
                    
                    for i in range(len(prices_numeric)):
                        if i == 0:
                            # Perubahan dari harga saat ini
                            daily_change = ((prices_numeric[i] - current_price) / current_price) * 100
                            daily_changes.append(daily_change)
                            cumulative_changes.append(daily_change)
                        else:
                            # Perubahan dari hari sebelumnya
                            daily_change = ((prices_numeric[i] - prices_numeric[i-1]) / prices_numeric[i-1]) * 100
                            daily_changes.append(daily_change)
                            cumulative_change = ((prices_numeric[i] - current_price) / current_price) * 100
                            cumulative_changes.append(cumulative_change)
                    
                    # Format tabel
                    display_df['Forecast_Price'] = display_df['Forecast_Price'].apply(lambda x: f"${x:,.2f}")
                    display_df['Daily_Change'] = [f"{x:+.2f}%" for x in daily_changes]
                    display_df['Cumulative_Change'] = [f"{x:+.2f}%" for x in cumulative_changes]
                    
                    # Reorder columns
                    display_df = display_df[['Day', 'Date', 'Forecast_Price', 'Daily_Change', 'Cumulative_Change']]
                    
                    # Style tabel
                    def color_change(val):
                        try:
                            change = float(val.strip('%'))
                            if change > 0:
                                return 'color: green'
                            elif change < 0:
                                return 'color: red'
                        except:
                            pass
                        return ''
                    
                    styled_df = display_df.style.applymap(color_change, subset=['Daily_Change', 'Cumulative_Change'])
                    
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                    
                    # 4. Download Forecast Data
                    csv = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv,
                        file_name=f"{symbol}_forecast_{forecast_days}days.csv",
                        mime="text/csv",
                    )
                    
                    # 5. Advanced Metrics
                    if show_metrics:
                        with st.expander("📊 Advanced Forecast Metrics"):
                            col_met1, col_met2 = st.columns(2)
                            
                            with col_met1:
                                st.markdown("##### 📈 Price Statistics")
                                st.write(f"**Minimum Price:** ${metrics.get('min_price', 0):,.2f}")
                                st.write(f"**Maximum Price:** ${metrics.get('max_price', 0):,.2f}")
                                st.write(f"**Price Range:** ${metrics.get('max_price', 0) - metrics.get('min_price', 0):,.2f}")
                                st.write(f"**Forecast Volatility:** {metrics.get('volatility', 0):.2f}%")
                            
                            with col_met2:
                                st.markdown("##### 📊 Model Performance")
                                st.write(f"**Window Size:** {window_size}")
                                st.write(f"**Features Used:** {len(features_used)}")
                                
                                # Berdasarkan trend, berikan rekomendasi
                                trend = metrics.get('trend', 'FLAT')
                                total_change = metrics.get('total_change_pct', 0)
                                
                                if trend == 'UP' and total_change > 5:
                                    st.success("**🎯 Recommendation:** Strong bullish trend. Consider accumulating on dips.")
                                elif trend == 'UP':
                                    st.info("**🎯 Recommendation:** Mild bullish trend. Consider holding existing positions.")
                                elif trend == 'DOWN' and total_change < -5:
                                    st.error("**🎯 Recommendation:** Strong bearish trend. Consider taking profits or hedging.")
                                elif trend == 'DOWN':
                                    st.warning("**🎯 Recommendation:** Mild bearish trend. Consider reducing exposure.")
                                else:
                                    st.info("**🎯 Recommendation:** Sideways trend. Consider range trading strategies.")
                    
                    # 6. Risk Warning
                    st.markdown("---")
                    st.warning("""
                    ⚠️ **Risk Disclaimer:**
                    - Forecasts are based on historical patterns and mathematical models
                    - Actual market movements may differ significantly
                    - Past performance is not indicative of future results
                    - Always conduct your own research and consult financial advisors
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Error dalam forecasting: {str(e)}")
                    with st.expander("🔍 Error Details"):
                        import traceback
                        st.code(traceback.format_exc())
        else:
            # Show preview before running
            st.info("👆 Klik **'Run Future Forecast'** untuk memulai prediksi")
            
            # Tampilkan info model
            st.markdown("### 📋 Model Information")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.write("**Model Type**")
                st.write("LSTM (Pre-trained)")
            
            with col_info2:
                st.write("**Forecast Range**")
                st.write(f"Up to {forecast_days} days")
            
            with col_info3:
                st.write("**Historical Data**")
                st.write(f"{len(df)} days available")
            
            # Preview data
            if 'close' in df.columns or 'Close' in df.columns:
                st.markdown("### 📊 Historical Data Preview")
                
                # Ambil data close terakhir 30 hari
                close_col = 'close' if 'close' in df.columns else 'Close'
                preview_data = df[[close_col]].tail(30).copy()
                
                fig_preview = go.Figure()
                fig_preview.add_trace(go.Scatter(
                    x=preview_data.index,
                    y=preview_data[close_col],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ))
                
                fig_preview.update_layout(
                    title='Last 30 Days (For Forecasting Context)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=300
                )
                
                st.plotly_chart(fig_preview, use_container_width=True)