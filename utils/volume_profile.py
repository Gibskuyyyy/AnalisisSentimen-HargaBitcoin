# streamlit_app/utils/volume_profile.py
import numpy as np
import pandas as pd

def compute_volume_profile(df, price_col='close', vol_col='volume', bins=30):
    prices = df[price_col].values
    vols = df[vol_col].values
    if len(prices) == 0:
        return pd.DataFrame(columns=['price_left','price_right','volume'])
    edges = np.linspace(prices.min(), prices.max(), bins+1)
    vol_sum = []
    for i in range(bins):
        mask = (prices >= edges[i]) & (prices < edges[i+1])
        vol_sum.append(vols[mask].sum())
    vp = pd.DataFrame({'price_left': edges[:-1], 'price_right': edges[1:], 'volume': vol_sum})
    return vp
