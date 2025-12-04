import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

# ============================================
# LOAD ENV
# ============================================
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "btc_price")

# ============================================
# FETCH NEW BTC PRICE
# ============================================
def fetch_latest_btc():
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=3)  # safety margin needed

    df = yf.download(
        "BTC-USD",
        start=yesterday.strftime("%Y-%m-%d"),
        interval="1d"
    )

    if df.empty:
        print("No data from Yahoo Finance.")
        return None

    # Remove multiindex column if exists (Yahoo sometimes returns this!)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Standardize the column names
    df.rename(
        columns={
            "Date": "time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        },
        inplace=True
    )

    df["time"] = pd.to_datetime(df["time"])

    # Drop any row with invalid time
    df = df.dropna(subset=["time"])

    return df


# ============================================
# CHECK IF DATE ALREADY EXISTS
# (uses tag "date")
# ============================================
def check_existing_date(client, date_str):
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: -30d)
            |> filter(fn: (r) => r["_measurement"] == "btc_price")
            |> filter(fn: (r) => r["date"] == "{date_str}")
            |> limit(n: 1)
    '''

    result = client.query_api().query(query, org=INFLUX_ORG)

    return len(result) > 0


# ============================================
# WRITE DATA TO INFLUXDB
# ============================================
def write_to_influx(df):
    with InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    ) as client:

        write_api = client.write_api(write_options=SYNCHRONOUS)

        for _, row in df.iterrows():

            # Make sure time is valid scalar
            time_value = pd.to_datetime(row["time"], errors="coerce")
            if pd.isna(time_value):
                print("[SKIP] invalid datetime:", row["time"])
                continue

            date_str = time_value.strftime("%Y-%m-%d")

            # Skip if data already exists
            if check_existing_date(client, date_str):
                print(f"[SKIP] {date_str} already exists in InfluxDB.")
                continue

            # Build point
            point = (
                Point("btc_price")
                .tag("symbol", "BTC-USD")
                .tag("date", date_str)
                .field("open", float(row["open"]))
                .field("high", float(row["high"]))
                .field("low", float(row["low"]))
                .field("close", float(row["close"]))
                .field("volume", float(row["volume"]))
                .time(time_value, WritePrecision.NS)
            )

            # Write to influx
            write_api.write(bucket=INFLUX_BUCKET, record=point)
            print(f"[WRITE] {date_str} wrote successfully")


# ============================================
# MAIN
# ============================================
def main():
    print("=== Fetching BTC price data ===")
    df = fetch_latest_btc()

    if df is None:
        return

    print(df)
    print(df.dtypes)

    print("=== Writing to InfluxDB ===")
    write_to_influx(df)

    print("=== DONE ===")


if __name__ == "__main__":
    main()