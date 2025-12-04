#!/usr/bin/env python3
"""
etl/tes.py - TEST PREDICTION DENGAN MEASUREMENT BARU
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from tensorflow.keras.models import load_model
import yfinance as yf
import ta
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Config
# ---------------------------
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
BUCKET = "sentiment_daily"

# Model tetap sama
MODEL_PATH = "models/best_lstm_model.h5"
SCALER_PATH = "models/minmax_scaler.pkl"
META_PATH = "models/model_metadata.json"

# 🔥 MEASUREMENT BARU
MEASUREMENT_NAME = "bitcoin_price_pred_new"

def ensure_float(value):
    if isinstance(value, (int, np.integer)):
        return float(value) + 0.0
    return float(value)

def fetch_data_from_yahoo(days=200):
    """Ambil data dari Yahoo Finance"""
    print(f"📥 Mengambil {days} hari data dari Yahoo Finance...")
    
    try:
        btc = yf.download(
            "BTC-USD", 
            period=f"{days}d",
            auto_adjust=False,
            progress=False
        )
        
        if btc.empty:
            return pd.DataFrame()
        
        # Flatten MultiIndex columns
        btc = btc.reset_index()
        new_cols = []
        for col in btc.columns:
            if col == 'Date':
                new_cols.append('Date')
            else:
                new_cols.append(col[0])
        btc.columns = new_cols
        
        btc['Date'] = pd.to_datetime(btc['Date'])
        btc = btc.set_index('Date').sort_index()
        btc = btc[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✅ Data: {len(btc)} baris, terakhir: {btc.index[-1].date()}")
        print(f"   Close: ${btc['Close'].iloc[-1]:,.2f}")
        
        return btc
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return pd.DataFrame()

def compute_features(df, features_list):
    """Hitung indikator teknis"""
    df2 = df.copy()
    
    # Mapping dari nama fitur ke fungsi perhitungan
    feature_calculations = {
        # Basic OHLCV
        'Open': lambda df: df['Open'],
        'High': lambda df: df['High'],
        'Low': lambda df: df['Low'],
        'Close': lambda df: df['Close'],
        'Volume': lambda df: df['Volume'],
        
        # Moving Averages
        'SMA_10': lambda df: ta.trend.sma_indicator(df['Close'], window=10),
        'SMA_50': lambda df: ta.trend.sma_indicator(df['Close'], window=50),
        'EMA_12': lambda df: ta.trend.ema_indicator(df['Close'], window=12),
        'EMA_26': lambda df: ta.trend.ema_indicator(df['Close'], window=26),
        
        # Momentum
        'RSI_14': lambda df: ta.momentum.rsi(df['Close'], window=14),
        
        # MACD
        'MACD': lambda df: ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd(),
        'MACD_signal': lambda df: ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_signal(),
        'MACD_hist': lambda df: ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd() - 
                                ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_signal(),
        
        # Volatility
        'ATR_14': lambda df: ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14),
        
        # Bollinger Bands
        'BB_middle': lambda df: ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_mavg(),
        'BB_high': lambda df: ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_hband(),
        'BB_low': lambda df: ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_lband(),
        'BB_width': lambda df: (ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_hband() - 
                                ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_lband()) / 
                                ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_mavg(),
        
        # Volume
        'OBV': lambda df: ta.volume.on_balance_volume(df['Close'], df['Volume']),
        'OBV_chg': lambda df: ta.volume.on_balance_volume(df['Close'], df['Volume']).pct_change(),
        
        # Oscillators
        'STOCH': lambda df: ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3),
        'WILLR': lambda df: ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14),
        
        # Returns
        'ret_1d': lambda df: df['Close'].pct_change(),
        'log_ret_1d': lambda df: np.log(df['Close']).diff(),
    }
    
    # Hitung hanya fitur yang ada di features_list dan belum ada di df2
    for feature in features_list:
        if feature not in df2.columns and feature in feature_calculations:
            try:
                df2[feature] = feature_calculations[feature](df2)
            except Exception as e:
                print(f"⚠ Error menghitung {feature}: {e}")
                df2[feature] = 0.0
        elif feature not in df2.columns:
            df2[feature] = 0.0
    
    # Drop NaN
    df2 = df2.dropna()
    
    return df2

def main():
    print("="*70)
    print(f"🤖 BTC PRICE PREDICTION - MEASUREMENT BARU: {MEASUREMENT_NAME}")
    print("="*70)
    print(f"📦 Model: {MODEL_PATH}")
    print(f"📦 Scaler: {SCALER_PATH}")
    print()
    
    # 1. Load metadata
    if not os.path.exists(META_PATH):
        print(f"❌ Metadata tidak ditemukan")
        return
    
    try:
        with open(META_PATH, 'r') as f:
            meta = json.load(f)
        
        features = meta.get('features_used', [])
        window_size = int(meta.get('best_parameters', {}).get('window_size', 14))
        model_version = meta.get('version', '1.0')
        
        print(f"✅ Model Version: {model_version}")
        print(f"✅ Features: {len(features)}, Window: {window_size}")
        
    except Exception as e:
        print(f"❌ Error metadata: {e}")
        return
    
    # 2. Load model & scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("❌ Model/scaler tidak ditemukan")
        return
    
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Model & scaler loaded")
    except Exception as e:
        print(f"❌ Error load model: {e}")
        return
    
    # 3. Get data
    min_days_needed = max(50 + window_size + 30, 200)
    df_raw = fetch_data_from_yahoo(days=min_days_needed)
    
    if df_raw.empty:
        print(f"❌ Data kosong")
        return
    
    # 4. Compute features
    print(f"🧮 Menghitung {len(features)} fitur...")
    df_feat = compute_features(df_raw, features)
    
    print(f"✅ Setelah perhitungan fitur: {len(df_feat)} baris tersisa")
    
    if len(df_feat) < window_size:
        print(f"❌ Data tidak cukup untuk window {window_size}")
        return
    
    # 5. Pastikan urutan kolom sesuai dengan features
    for feature in features:
        if feature not in df_feat.columns:
            df_feat[feature] = 0.0
    
    df_feat = df_feat[features]
    
    # 6. Prepare data for prediction
    X_data = df_feat.tail(window_size)
    
    print(f"📊 Data untuk prediksi:")
    print(f"   Shape: {X_data.shape}")
    print(f"   Periode: {X_data.index[0].date()} hingga {X_data.index[-1].date()}")
    
    # 7. Scale and predict
    try:
        X_scaled = scaler.transform(X_data)
        X_reshaped = X_scaled.reshape(1, window_size, len(features))
        
        pred_scaled = float(model.predict(X_reshaped, verbose=0)[0,0])
        actual_price = float(df_raw['Close'].iloc[-1])
        
        # Inverse transform
        if 'Close' in features:
            close_idx = features.index('Close')
            dummy = np.zeros((1, len(features)))
            dummy[0, close_idx] = pred_scaled
            dummy_inverse = scaler.inverse_transform(dummy)
            prediction = float(dummy_inverse[0, close_idx])
        else:
            prediction = pred_scaled
        
        print(f"✅ Predicted: ${prediction:,.2f}")
        print(f"✅ Actual: ${actual_price:,.2f}")
        
        # Validate
        change_pct = ((prediction - actual_price) / actual_price) * 100
        print(f"✅ Change: {change_pct:+.2f}%")
        
        # Fallback jika prediksi tidak realistis
        if abs(change_pct) > 50 or prediction < 1000:
            print(f"⚠ Prediction unrealistic, using fallback")
            fallback = df_raw['Close'].tail(7).mean() * 1.005
            prediction = fallback
            change_pct = ((prediction - actual_price) / actual_price) * 100
            print(f"✅ Fallback: ${prediction:,.2f} ({change_pct:+.2f}%)")
        
    except Exception as e:
        print(f"❌ Error prediction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 8. Write to InfluxDB
    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG,
        timeout=60_000
    )
    
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    # Waktu untuk besok
    tomorrow = datetime.utcnow() + timedelta(days=1)
    
    # 🔥 GUNAKAN MEASUREMENT BARU DI SINI!
    point = Point(MEASUREMENT_NAME) \
        .tag("asset", "BTC") \
        .tag("model", "GA_LSTM") \
        .tag("version", model_version) \
        .field("pred_close", ensure_float(prediction)) \
        .field("actual", ensure_float(actual_price)) \
        .field("abs_error", ensure_float(abs(prediction - actual_price))) \
        .field("change_pct", ensure_float(change_pct)) \
        .time(tomorrow)
    
    try:
        write_api.write(bucket=BUCKET, org=INFLUX_ORG, record=point)
        print(f"\n💾 💾 💾 TERSIMPAN KE MEASUREMENT BARU: {MEASUREMENT_NAME}")
        print(f"   📅 Untuk tanggal: {tomorrow.date()}")
        print(f"   💰 Prediksi: ${prediction:,.2f} ({change_pct:+.2f}%)")
        print(f"   🏷️  Tags: model=GA_LSTM, version={model_version}")
        
    except Exception as e:
        print(f"❌ Error saving: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()
    
    print(f"\n✨ SEMUA DATA BARU AKAN MASUK KE: {MEASUREMENT_NAME}")

if __name__ == "__main__":
    main()