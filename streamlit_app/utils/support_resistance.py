import numpy as np
from scipy.signal import find_peaks

def find_support_resistance(df, n=6):
    # Pastikan hanya ambil close sebagai 1D numpy array
    prices = df['close'].to_numpy().astype(float).flatten()

    # Safety check: minimal 10 data
    if len(prices) < 10:
        return [], []

    # Temukan resistance (puncak)
    peaks, _ = find_peaks(prices, distance=n)
    # Temukan support (lembah)
    troughs, _ = find_peaks(-prices, distance=n)

    sup_levels = prices[troughs]
    res_levels = prices[peaks]

    return sup_levels, res_levels
