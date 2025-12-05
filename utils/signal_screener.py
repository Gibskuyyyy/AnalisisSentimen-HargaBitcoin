# utils/signal_screener.py
import numpy as np
import pandas as pd

def _to_1d_series(out: pd.DataFrame, col: str) -> pd.Series:
    """
    Return a 1-D numeric Series aligned to out.index.
    Handles if out[col] is missing, DataFrame, Series with weird shapes, etc.
    """
    if col not in out.columns:
        s = pd.Series(index=out.index, dtype=float)
        return s

    s = out[col]

    # If it's a DataFrame, take first column (best-effort)
    if isinstance(s, pd.DataFrame):
        if s.shape[1] >= 1:
            s = s.iloc[:, 0]
        else:
            s = s.squeeze()

    # Now coercion to numeric Series
    s = pd.Series(s).squeeze()  # convert ndarray -> Series if needed
    s = pd.to_numeric(s, errors="coerce")

    # Ensure index is exactly out.index
    try:
        s.index = out.index
    except Exception:
        # fallback: create new Series from values with out.index
        s = pd.Series(s.values.flatten(), index=out.index)

    # flatten to 1-D values and return Series
    values = s.values.reshape(-1)
    return pd.Series(values, index=out.index, dtype=float)


def add_screener_signals(df: pd.DataFrame, rsi_buy=30, rsi_sell=70) -> pd.DataFrame:
    """
    Robust screener: forces required columns to 1-D numeric aligned Series,
    computes boolean signals using numpy (no pandas alignment surprises).
    """
    out = df.copy()

    if out is None:
        out = pd.DataFrame()

    if out.empty:
        # return empty but with signal columns
        out = out.copy()
        out['signal_buy'] = pd.Series(dtype=bool)
        out['signal_sell'] = pd.Series(dtype=bool)
        return out

    required = ['close', 'RSI_14', 'MACD', 'BB_low', 'BB_high']

    # Build aligned Series for each required column
    series_map = {}
    for col in required:
        series_map[col] = _to_1d_series(out, col)

    # Convert to numpy arrays
    close_arr = series_map['close'].to_numpy(dtype=float)
    rsi_arr = series_map['RSI_14'].to_numpy(dtype=float)
    macd_arr = series_map['MACD'].to_numpy(dtype=float)
    bb_low_arr = series_map['BB_low'].to_numpy(dtype=float)
    bb_high_arr = series_map['BB_high'].to_numpy(dtype=float)

    # Ensure same length
    n = len(out.index)
    for arr in (close_arr, rsi_arr, macd_arr, bb_low_arr, bb_high_arr):
        if len(arr) != n:
            # reindex fallback: pad/truncate to length n
            arr = np.resize(arr, n)

    # Safe elementwise comparisons using numpy (NaN -> False)
    def lt_safe(a, b):
        # returns boolean array where a < b, treats NaN as False
        with np.errstate(invalid='ignore'):
            res = a < b
        res = np.where(np.isnan(res), False, res)
        return res

    def gt_safe(a, b):
        with np.errstate(invalid='ignore'):
            res = a > b
        res = np.where(np.isnan(res), False, res)
        return res

    buy_cond = np.logical_or(lt_safe(rsi_arr, rsi_buy), lt_safe(close_arr, bb_low_arr))
    sell_cond = np.logical_or(gt_safe(rsi_arr, rsi_sell), gt_safe(close_arr, bb_high_arr))

    # Create boolean Series and attach to DataFrame
    out['signal_buy'] = pd.Series(buy_cond.astype(bool), index=out.index)
    out['signal_sell'] = pd.Series(sell_cond.astype(bool), index=out.index)

    return out
