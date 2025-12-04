# utils/influx_helper.py - VERSI FIXED
import os
import pandas as pd
from influxdb_client import InfluxDBClient
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

load_dotenv()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
BUCKET = os.getenv("INFLUX_BUCKET_OHLCV", "sentiment_daily")


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

def validate_dataframe_columns(df, required_cols=['close']):
    """Validasi bahwa DataFrame memiliki kolom yang diperlukan."""
    df = clean_duplicate_columns(df.copy())
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Coba temukan dengan variasi nama
        for col in missing_cols:
            found = False
            for variation in [col.capitalize(), col.upper(), col.title()]:
                if variation in df.columns:
                    df[col] = df[variation]
                    found = True
                    print(f"Found {col} as {variation}")
                    break
            
            if not found:
                print(f"Cannot find column {col}, creating with NaN")
                df[col] = np.nan
    
    return df

def clean_duplicate_columns(df):
    """Hapus kolom duplikat, simpan yang pertama."""
    if isinstance(df, pd.DataFrame):
        df = df.loc[:, ~df.columns.duplicated()]
    return df

# ============================================================
# INTERNAL QUERY WRAPPER
# ============================================================
def _qdf(query: str) -> pd.DataFrame:
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    qapi = client.query_api()
    try:
        df = qapi.query_data_frame(query)
        if isinstance(df, list):
            df = pd.concat(df, ignore_index=True)
        return df
    except Exception as e:
        print("Influx query error:", e)
        return pd.DataFrame()
    finally:
        client.close()


def _remove_timezone(df):
    """Remove timezone from datetime columns if present"""
    if df.empty:
        return df
    
    # Cek semua kolom datetime dengan pendekatan yang lebih aman
    for col in df.columns:
        try:
            # Cek jika kolom ini datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Coba hapus timezone
                if hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                    try:
                        df[col] = df[col].dt.tz_convert(None)
                    except:
                        try:
                            df[col] = df[col].dt.tz_localize(None)
                        except:
                            pass
        except:
            continue
    
    return df


# ============================================================
# 1️⃣ PRICE HISTORY & LATEST
# ============================================================
def get_price_history(days=365):
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r._measurement == "btc_price")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns: ["_time"])
    '''
    df = _qdf(flux)
    if df.empty:
        return pd.DataFrame()
    
    if '_time' in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = _remove_timezone(df)
        return df.set_index("_time")
    else:
        return df


def get_latest_price():
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -3d)
      |> filter(fn: (r) => r["_measurement"] == "btc_price")
      |> last()
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
    '''
    df = _qdf(flux)
    if df.empty:
        return None
    
    df = _remove_timezone(df)
    if df.empty:
        return None
    
    row = df.iloc[-1]
    return {"close": float(row.get("close", 0)), "time": row.get("_time", datetime.now())}


# ============================================================
# 2️⃣ OHLCV HISTORY
# ============================================================
def get_ohlcv_history(days=365):
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r["_measurement"] == "bitcoin_ohlcv")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns:["_time"])
    '''
    df = _qdf(flux)
    if df.empty:
        return pd.DataFrame()
    
    if '_time' in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = _remove_timezone(df)
        return df.set_index("_time")
    else:
        return df


# ============================================================
# 3️⃣ SENTIMENT HISTORY & LATEST
# ============================================================
def get_sentiment_history(days=30):
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r._measurement == "bitcoin_sentiment" or r._measurement == "btc_sentiment")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns:["_time"])
    '''
    df = _qdf(flux)
    if df.empty:
        return pd.DataFrame()
    
    if '_time' in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = _remove_timezone(df)
    
    return df


def get_latest_sentiment():
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -7d)
      |> filter(fn: (r) => r._measurement == "bitcoin_sentiment")
      |> last()
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
    '''
    df = _qdf(flux)
    if df.empty:
        return None
    
    df = _remove_timezone(df)
    if df.empty:
        return None
    
    row = df.iloc[-1]
    return {
        "positive": float(row.get("positive", 0)),
        "neutral": float(row.get("neutral", 0)),
        "negative": float(row.get("negative", 0)),
        "time": row.get("_time", datetime.now())
    }


# ============================================================
# 4️⃣ PREDICTION HISTORY & LATEST
# ============================================================
def get_latest_forecast():
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "btc_price_pred")
      |> last()
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
    '''
    df = _qdf(flux)
    if df.empty:
        return None
    
    df = _remove_timezone(df)
    if df.empty:
        return None
    
    row = df.iloc[-1]
    return {
        "pred_close": float(row.get("pred_close", 0)),
        "actual": float(row.get("actual", 0)),
        "abs_error": float(row.get("abs_error", 0)),
        "time": row.get("_time", datetime.now())
    }


def get_prediction_history(days=30):
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r["_measurement"] == "btc_price_pred")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns:["_time"])
    '''
    df = _qdf(flux)
    if df.empty:
        return pd.DataFrame()
    
    if '_time' in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = _remove_timezone(df)
        return df.set_index("_time")
    else:
        return df


def get_prediction_data_raw(days=7):
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r["_measurement"] == "btc_price_pred")
      |> sort(columns: ["_time"], desc: true)
    '''
    df = _qdf(flux)
    if df.empty:
        return pd.DataFrame()
    
    if '_time' in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = _remove_timezone(df)
    
    return df


# ============================================================
# 5️⃣ NEWS RAW
# ============================================================
def get_news_raw(days=7):
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r["_measurement"] == "news_raw")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns:["_time"], desc: true)
    '''
    df = _qdf(flux)
    if df.empty:
        return pd.DataFrame()
    
    if '_time' in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = _remove_timezone(df)
    
    return df


# ============================================================
# 6️⃣ UTILITY: CHECK MEASUREMENTS
# ============================================================
def check_measurements():
    flux = f'''
    import "influxdata/influxdb/schema"
    schema.measurements(bucket: "{BUCKET}")
    '''
    df = _qdf(flux)
    if df.empty:
        return []
    return df['_value'].tolist()


def check_latest_data():
    """Check latest data in all measurements"""
    print("="*70)
    print("🔍 CHECKING LATEST DATA IN INFLUXDB")
    print("="*70)
    
    measurements = check_measurements()
    
    for measurement in measurements:
        try:
            query = f'''
            from(bucket:"{BUCKET}")
              |> range(start: -2d)
              |> filter(fn: (r) => r._measurement == "{measurement}")
              |> sort(columns: ["_time"], desc: true)
              |> limit(n: 1)
            '''
            
            client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
            result = client.query_api().query(query)
            
            latest_time = None
            for table in result:
                for record in table.records:
                    latest_time = record.get_time()
                    break
                break
            
            if latest_time:
                print(f"📊 {measurement}: {latest_time}")
            else:
                print(f"📊 {measurement}: No recent data")
                
            client.close()
            
        except Exception as e:
            print(f"📊 {measurement}: Error - {e}")
    
    print("="*70)


def get_measurement_stats(measurement_name):
    """Get statistics for a specific measurement"""
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "{measurement_name}")
      |> count()
      |> group()
      |> sum()
    '''
    
    df = _qdf(flux)
    if df.empty:
        return 0
    
    return df['_value'].iloc[0] if '_value' in df.columns else 0


# ============================================================
# 7️⃣ BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================
def get_forecast_history(days=7):
    """Alias untuk backward compatibility"""
    return get_prediction_history(days)


def get_price_pred_new(days=7):
    """Get data from bitcoin_price_pred_new measurement"""
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r._measurement == "bitcoin_price_pred_new")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns:["_time"], desc: true)
    '''
    df = _qdf(flux)
    if df.empty:
        return pd.DataFrame()
    
    if '_time' in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = _remove_timezone(df)
    
    return df


def get_bitcoin_ohlcv(days=30):
    """Alias untuk bitcoin_ohlcv"""
    return get_ohlcv_history(days)


def get_btc_sentiment(days=30):
    """Get data from btc_sentiment measurement"""
    flux = f'''
    from(bucket:"{BUCKET}")
      |> range(start: -{days}d)
      |> filter(fn: (r) => r._measurement == "btc_sentiment")
      |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
      |> sort(columns:["_time"])
    '''
    df = _qdf(flux)
    if df.empty:
        return pd.DataFrame()
    
    if '_time' in df.columns:
        df["_time"] = pd.to_datetime(df["_time"])
        df = _remove_timezone(df)
    
    return df


# ============================================================
# 8️⃣ TESTING
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("🧪 TESTING INFLUX HELPER")
    print("="*70)

    print("🔗 URL:", INFLUX_URL)
    print("📦 Bucket:", BUCKET)
    print("🏢 Org:", INFLUX_ORG)

    print("\n🔍 Latest forecast:")
    print(get_latest_forecast())

    print("\n🔍 Latest sentiment:")
    print(get_latest_sentiment())

    print("\n📊 Sentiment history (testing):")
    sent = get_sentiment_history(7)
    if not sent.empty:
        print(f"Shape: {sent.shape}")
        print(f"Columns: {sent.columns.tolist()}")
        if '_time' in sent.columns:
            print(f"_time dtype: {sent['_time'].dtype}")
            print(f"Sample times: {sent['_time'].head(3).tolist()}")
    
    print("\n📋 Available measurements:")
    measurements = check_measurements()
    for m in measurements:
        print(f"  • {m}")
    
    print("\n✅ Testing completed")