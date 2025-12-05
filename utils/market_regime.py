# streamlit_app/utils/market_regime.py
import pandas as pd
import numpy as np

def detect_market_regime(df):
    out = df.copy()
    if 'SMA_10' in df.columns and 'SMA_50' in df.columns:
        s_short = df['SMA_10']; s_long = df['SMA_50']
    else:
        s_short = df['close'].rolling(10).mean(); s_long = df['close'].rolling(50).mean()
    slope = s_short.diff()
    regime = pd.Series(index=df.index, dtype='object')
    regime[:] = 'neutral'
    regime[(s_short > s_long) & (slope > 0)] = 'bull'
    regime[(s_short < s_long) & (slope < 0)] = 'bear'
    out['regime'] = regime
    out['regime_score'] = ((s_short - s_long) / s_long).fillna(0)
    return out
