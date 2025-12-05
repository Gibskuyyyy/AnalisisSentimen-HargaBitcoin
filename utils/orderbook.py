# streamlit_app/utils/orderbook.py
import requests
import pandas as pd

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"

def fetch_binance_orderbook(symbol="BTCUSDT", limit=100):
    params = {"symbol": symbol, "limit": limit}
    try:
        r = requests.get(BINANCE_DEPTH_URL, params=params, timeout=8)
        if r.status_code != 200:
            return None
        j = r.json()
        bids = [(float(p[0]), float(p[1])) for p in j.get('bids',[])]
        asks = [(float(p[0]), float(p[1])) for p in j.get('asks',[])]
        # compute totals
        total_bids = sum([v for p,v in bids])
        total_asks = sum([v for p,v in asks])
        imbalance = (total_bids - total_asks) / (total_bids + total_asks) if (total_bids+total_asks)>0 else 0
        return {"bids": bids, "asks": asks, "total_bids": total_bids, "total_asks": total_asks, "imbalance": imbalance}
    except Exception as e:
        print("Orderbook fetch error:", e)
        return None
