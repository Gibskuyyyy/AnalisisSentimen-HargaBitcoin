import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from influxdb_client import InfluxDBClient, Point, WriteOptions
import ta
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import json

# ============================
# Load environment
# ============================
load_dotenv()
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = "sentiment_daily"  # tetap bucket ini
DATA_DIR = "data"
MODEL_DIR = "models"

# ============================
# Load model & scaler
# ============================
scaler = joblib.load(os.path.join(MODEL_DIR, "minmax_scaler.pkl"))
model = load_model(os.path.join(MODEL_DIR, "best_lstm_model.h5"))

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
    meta = json.load(f)

features_used = meta['features_used']
window_size = meta['best_parameters']['window_size']

print(f"✅ Model & scaler loaded, {len(features_used)} fitur, window={window_size}")

# ============================
# Fetch OHLCV dari Yahoo
# ============================
def fetch_yahoo_ohlcv(days=400):
    symbol = "BTC-USD"
    start = (pd.Timestamp.today() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
    if df.empty:
        print("❌ Data Yahoo Finance kosong")
        return pd.DataFrame()

    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df

# ============================
# Hitung indikator teknikal
# ============================
def compute_features(df_raw):
    df_ind = df_raw.copy()

    # Pastikan 1D Series
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

    df_feat = df_ind.dropna().copy()
    return df_feat


def create_rolling_windows(df_feat, window_size, feature_cols):
    X = []
    for i in range(window_size, len(df_feat)):
        X.append(df_feat[feature_cols].iloc[i-window_size:i].values)
    return np.array(X)  # shape = (num_samples, window_size, num_features)

# ============================
# Menulis prediksi ke InfluxDB
# ============================
def write_prediction_influx(df_pred, influx_url, influx_token, influx_org, influx_bucket):
    # pastikan index datetime tz-aware (UTC)
    if df_pred.index.tzinfo is None or df_pred.index.tz is None:
        df_pred.index = df_pred.index.tz_localize("UTC")

    client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
    write_api = client.write_api(write_options=WriteOptions(batch_size=1, flush_interval=1000, jitter_interval=0))

    sukses = 0
    gagal = 0

    df_pred.loc[:, 'pred_close'] = df_pred['pred_close'].astype(float)
    df_pred.loc[:, 'actual'] = df_pred['Close'].astype(float)

    for t, row in df_pred.iterrows():
        try:
            pred_val = row['pred_close']
            actual_val = row['actual']

            if isinstance(pred_val, pd.Series): pred_val = pred_val.iloc[0]
            if isinstance(actual_val, pd.Series): actual_val = actual_val.iloc[0]

            pt = (
                Point("btc_price_pred")
                .tag("asset", "BTC")
                .field("pred_close", float(pred_val))
                .field("actual", float(actual_val))
                .field("abs_error", abs(float(pred_val) - float(actual_val)))
                .time(t.to_pydatetime())
            )
            write_api.write(bucket=influx_bucket, org=influx_org, record=pt)
            sukses += 1
        except Exception as e:
            print(f"⚠️ Error menulis timestamp {t}: {e}")
            gagal += 1

    write_api.flush()
    write_api.close()
    client.close()

    print(f"✅ {sukses} prediksi berhasil ditulis ke InfluxDB.")
    if gagal > 0:
        print(f"⚠️ {gagal} prediksi gagal ditulis.")

def main():
    print("📈 Mengambil data harga BTC...")
    df_raw = fetch_yahoo_ohlcv(days=400)
    if df_raw.empty:
        return

    print("⚙️ Menghitung indikator teknikal...")
    df_feat = compute_features(df_raw)
    feature_cols = features_used

    print("🔄 Membuat rolling windows...")
    X = create_rolling_windows(df_feat, window_size, feature_cols)

    print("🔄 Scaling fitur...")
    X_scaled = scaler.transform(X.reshape(-1, len(feature_cols))).reshape(X.shape)

    print("🤖 Prediksi harga...")
    preds_scaled = model.predict(X_scaled).flatten()

    # ============================
    # Inverse transform ke harga asli
    # ============================
    X_dummy = np.zeros((len(preds_scaled), len(features_used)))
    close_idx = features_used.index('Close')
    X_dummy[:, close_idx] = preds_scaled
    preds_actual = scaler.inverse_transform(X_dummy)[:, close_idx]

    # ============================
    # Buat DataFrame prediksi
    # ============================
    df_pred = df_feat.iloc[window_size:].copy()
    df_pred.loc[:, 'pred_close'] = preds_actual
    df_pred.loc[:, 'actual'] = df_pred['Close']

    # Pastikan index tz-aware (UTC) untuk InfluxDB
    df_pred.index = df_pred.index.tz_localize('UTC')

    # Batasi prediksi sesuai retention (misal 30 hari terakhir)
    retention_start = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)
    df_pred_to_write = df_pred[df_pred.index >= retention_start]

    print("💾 Menyimpan hasil prediksi ke InfluxDB...")
    write_prediction_influx(
        df_pred=df_pred_to_write,
        influx_url=INFLUX_URL,
        influx_token=INFLUX_TOKEN,
        influx_org=INFLUX_ORG,
        influx_bucket=INFLUX_BUCKET
    )

    print("\n📊 5 Prediksi Terakhir:")
    print(df_pred[['pred_close','actual']].tail(5))

    print("✅ Selesai!")

if __name__ == "__main__":
    main()
