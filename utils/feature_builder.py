import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler

def clean_duplicate_columns(df):
    """Hapus kolom duplikat, simpan yang pertama."""
    if isinstance(df, pd.DataFrame):
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def compute_features_update_style(df_raw):
    """
    Hitung indikator persis seperti di update_prediction.py
    """
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
    
    # Fill NaN
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

# Di utils/feature_builder.py, tambahkan di bagian akhir:

def ensure_ohlcv_columns(df):
    """Memastikan kolom OHLCV ada dengan nama yang benar"""
    df = df.copy()
    
    # Mapping nama kolom
    column_mapping = {
        'open': ['open', 'Open', 'OPEN'],
        'high': ['high', 'High', 'HIGH'],
        'low': ['low', 'Low', 'LOW'],
        'close': ['close', 'Close', 'CLOSE'],
        'volume': ['volume', 'Volume', 'VOLUME']
    }
    
    for target, possible_names in column_mapping.items():
        if target not in df.columns:
            for name in possible_names:
                if name in df.columns:
                    df[target] = df[name]
                    break
            else:
                # Jika tidak ditemukan, buat kolom dengan NaN
                df[target] = np.nan
    
    return df