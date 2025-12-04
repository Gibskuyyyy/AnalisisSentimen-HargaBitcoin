# utils/indicators.py
import pandas as pd
import numpy as np
import ta

def add_indicators(df):
    """Tambah indikator teknikal dengan penanganan error yang lebih baik."""
    try:
        df = df.copy()
        
        # Pastikan index adalah DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: Index is not DatetimeIndex")
        
        # ============================
        # NORMALISASI KOLOM OHLCV
        # ============================
        # Pastikan kolom ada dan adalah Series
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                # Ambil data kolom
                col_data = df[c]
                
                # Handle berbagai tipe data
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]  # Ambil kolom pertama
                elif isinstance(col_data, np.ndarray):
                    col_data = pd.Series(col_data, index=df.index)
                elif not isinstance(col_data, pd.Series):
                    col_data = pd.Series(col_data, index=df.index)
                
                # Konversi ke numeric
                df[c] = pd.to_numeric(col_data, errors='coerce')
            else:
                # Coba cari lowercase version
                c_lower = c.lower()
                if c_lower in df.columns:
                    df[c] = pd.to_numeric(df[c_lower], errors='coerce')
                else:
                    df[c] = np.nan
        
        # Isi NaN
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = df[c].ffill().bfill()
        
        # ============================
        # HITUNG INDIKATOR HANYA JIKA ADA DATA YANG CUKUP
        # ============================
        if len(df) < 2:
            print("Not enough data for indicators")
            return df
        
        # 1. Moving Averages
        try:
            df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
            df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
            df["EMA_12"] = df["Close"].ewm(span=12, min_periods=1).mean()
            df["EMA_26"] = df["Close"].ewm(span=26, min_periods=1).mean()
        except Exception as e:
            print(f"Error calculating MAs: {e}")
        
        # 2. RSI
        try:
            if len(df) >= 14:
                rsi_indicator = ta.momentum.RSIIndicator(close=df["Close"], window=14)
                df["RSI_14"] = rsi_indicator.rsi()
            else:
                df["RSI_14"] = np.nan
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            df["RSI_14"] = np.nan
        
        # 3. MACD
        try:
            if len(df) >= 26:
                macd_indicator = ta.trend.MACD(close=df["Close"])
                df["MACD"] = macd_indicator.macd()
                df["MACD_signal"] = macd_indicator.macd_signal()
                df["MACD_hist"] = macd_indicator.macd_diff()
            else:
                df["MACD"] = df["MACD_signal"] = df["MACD_hist"] = np.nan
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            df["MACD"] = df["MACD_signal"] = df["MACD_hist"] = np.nan
        
        # 4. Bollinger Bands
        try:
            if len(df) >= 20:
                bb_indicator = ta.volatility.BollingerBands(close=df["Close"], window=20)
                df["BB_high"] = bb_indicator.bollinger_hband()
                df["BB_low"] = bb_indicator.bollinger_lband()
                df["BB_middle"] = bb_indicator.bollinger_mavg()
                df["BB_width"] = df["BB_high"] - df["BB_low"]
            else:
                df["BB_high"] = df["BB_low"] = df["BB_middle"] = df["BB_width"] = np.nan
        except Exception as e:
            print(f"Error calculating BB: {e}")
            df["BB_high"] = df["BB_low"] = df["BB_middle"] = df["BB_width"] = np.nan
        
        # Isi NaN untuk semua kolom baru
        for col in df.columns:
            if col not in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = df[col].ffill().bfill()
        
        return df
        
    except Exception as e:
        print(f"❌ Critical error in add_indicators: {e}")
        import traceback
        traceback.print_exc()
        return df  # Return df as-is without indicators
    

def calculate_all_indicators(df):
    """
    Calculate ALL 24 technical indicators needed for the model.
    """
    df = df.copy()
    
    # Pastikan kita memiliki kolom OHLCV dengan nama yang benar
    column_map = {
        'open': 'Open', 'Open': 'Open',
        'high': 'High', 'High': 'High',
        'low': 'Low', 'Low': 'Low',
        'close': 'Close', 'Close': 'Close',
        'volume': 'Volume', 'Volume': 'Volume'
    }
    
    for alt_name, std_name in column_map.items():
        if alt_name in df.columns and std_name not in df.columns:
            df[std_name] = df[alt_name]
    
    # Hitung indikator dasar yang sudah ada
    df = add_indicators(df)
    
    # Hitung indikator yang mungkin belum ada
    
    # 1. ATR (Average True Range)
    if 'ATR_14' not in df.columns:
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr = np.maximum(high - low, 
                       np.abs(high - close.shift(1)), 
                       np.abs(low - close.shift(1)))
        df['ATR_14'] = tr.rolling(window=14).mean()
    
    # 2. OBV (On Balance Volume)
    if 'OBV' not in df.columns and 'Volume' in df.columns:
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV'] = obv
    
    # 3. OBV Change
    if 'OBV_chg' not in df.columns and 'OBV' in df.columns:
        df['OBV_chg'] = df['OBV'].diff()
    
    # 4. Stochastic Oscillator
    if 'STOCH' not in df.columns:
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['STOCH'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    
    # 5. Williams %R
    if 'WILLR' not in df.columns:
        high_14 = df['High'].rolling(window=14).max()
        low_14 = df['Low'].rolling(window=14).min()
        df['WILLR'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    
    # 6. Daily Return
    if 'ret_1d' not in df.columns:
        df['ret_1d'] = df['Close'].pct_change()
    
    # 7. Log Return
    if 'log_ret_1d' not in df.columns:
        df['log_ret_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df