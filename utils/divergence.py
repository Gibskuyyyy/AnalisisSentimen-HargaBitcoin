# streamlit_app/utils/divergence.py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def detect_divergences(df, lookback=60, indicator='RSI_14'):
    if indicator not in df.columns:
        return None

    out = df.copy()

    prices = out['close'].astype(float).values.reshape(-1)
    ind = out[indicator].astype(float).values.reshape(-1)

    pk_idx, _ = find_peaks(prices, distance=5)
    tr_idx, _ = find_peaks(-prices, distance=5)

    if len(pk_idx) < 2 and len(tr_idx) < 2:
        return None

    bull = np.zeros(len(out), dtype=bool)
    bear = np.zeros(len(out), dtype=bool)

    # bullish divergence (harga lower low, indicator higher low)
    for i in range(len(tr_idx)-1):
        t1 = tr_idx[i]
        t2 = tr_idx[i+1]
        if prices[t2] < prices[t1] and ind[t2] > ind[t1]:
            bull[t2] = True

    # bearish divergence (harga higher high, indicator lower high)
    for i in range(len(pk_idx)-1):
        p1 = pk_idx[i]
        p2 = pk_idx[i+1]
        if prices[p2] > prices[p1] and ind[p2] < ind[p1]:
            bear[p2] = True

    out['bull_divergence'] = bull
    out['bear_divergence'] = bear
    return out
