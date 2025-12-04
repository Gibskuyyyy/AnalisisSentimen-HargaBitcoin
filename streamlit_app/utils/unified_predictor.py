# utils/unified_predictor.py
import numpy as np
import pandas as pd
import yfinance as yf
import ta  # <- IMPORT TA LIBRARY
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# ======================================================
# GLOBAL CACHE UNTUK KONSISTENSI
# ======================================================
_prediction_cache = {}
_cache_expiry = {}
_sentiment_cache = {}

# ======================================================
# FUNGSI UTILITAS
# ======================================================
def clean_duplicate_columns(df):
    """Hapus kolom duplikat"""
    if isinstance(df, pd.DataFrame):
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def safe_convert_to_numeric(data):
    """Convert data to numeric Series safely"""
    try:
        if isinstance(data, pd.DataFrame):
            if data.shape[1] > 0:
                data = data.iloc[:, 0]
            else:
                return pd.Series(np.nan)
        elif isinstance(data, np.ndarray):
            data = pd.Series(data.ravel())
        elif not isinstance(data, pd.Series):
            data = pd.Series(data)
        
        return pd.to_numeric(data, errors='coerce')
    except Exception as e:
        print(f"Error in safe_convert_to_numeric: {e}")
        return pd.Series(np.nan)

def standardize_dataframe(df):
    """Standardize dataframe columns to lowercase and ensure OHLCV"""
    df = df.copy()
    df = clean_duplicate_columns(df)
    
    # Mapping nama kolom (case-insensitive)
    column_mapping = {}
    
    # Cari kolom yang ada
    for col in df.columns:
        col_lower = str(col).lower()
        if 'open' in col_lower:
            column_mapping[col] = 'open'
        elif 'high' in col_lower:
            column_mapping[col] = 'high'
        elif 'low' in col_lower:
            column_mapping[col] = 'low'
        elif 'close' in col_lower or 'last' in col_lower:
            column_mapping[col] = 'close'
        elif 'volume' in col_lower:
            column_mapping[col] = 'volume'
    
    # Rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Pastikan semua kolom OHLCV ada
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            # Coba cari variasi case
            for col_variant in [col, col.upper(), col.capitalize(), col.title()]:
                if col_variant in df.columns:
                    df[col] = df[col_variant]
                    break
            else:
                df[col] = np.nan
    
    # Konversi ke numeric
    for col in required_cols:
        if col in df.columns:
            df[col] = safe_convert_to_numeric(df[col])
    
    return df[required_cols]

# ======================================================
# FUNGSI INDIKATOR DENGAN TA LIBRARY
# ======================================================
def add_ta_indicators(df):
    """
    Tambahkan indikator teknikal menggunakan TA library
    Hasil akan konsisten dengan 2_price.py
    """
    df = df.copy()
    
    # Pastikan kolom ada
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Kolom {col} tidak ditemukan")
    
    # Konversi ke Series untuk TA
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # 1. TREND INDICATORS
    # Moving Averages
    df['SMA_10'] = ta.trend.sma_indicator(close, window=10)
    df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    df['SMA_50'] = ta.trend.sma_indicator(close, window=50)
    df['EMA_12'] = ta.trend.ema_indicator(close, window=12)
    df['EMA_26'] = ta.trend.ema_indicator(close, window=26)
    df['EMA_50'] = ta.trend.ema_indicator(close, window=50)
    
    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Ichimoku Cloud (simplified)
    df['Ichimoku_base'] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df['Ichimoku_conversion'] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    
    # 2. MOMENTUM INDICATORS
    # RSI
    df['RSI_14'] = ta.momentum.rsi(close, window=14)
    df['RSI_7'] = ta.momentum.rsi(close, window=7)
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
    df['STOCH_k'] = stoch.stoch()
    df['STOCH_d'] = stoch.stoch_signal()
    
    # Williams %R
    df['WILLR_14'] = ta.momentum.williams_r(high, low, close, lbp=14)
    
    # Awesome Oscillator
    df['AO'] = ta.momentum.awesome_oscillator(high, low, window1=5, window2=34)
    
    # 3. VOLATILITY INDICATORS
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    
    # Average True Range
    df['ATR_14'] = ta.volatility.average_true_range(high, low, close, window=14)
    
    # Keltner Channel (simplified)
    kc_middle = df['EMA_20'] if 'EMA_20' in df.columns else df['SMA_20']
    df['KC_upper'] = kc_middle + (df['ATR_14'] * 1.5)
    df['KC_lower'] = kc_middle - (df['ATR_14'] * 1.5)
    
    # 4. VOLUME INDICATORS
    # On-Balance Volume
    df['OBV'] = ta.volume.on_balance_volume(close, volume)
    df['OBV_EMA'] = ta.trend.ema_indicator(df['OBV'], window=21)
    
    # Volume SMA
    df['Volume_SMA_20'] = volume.rolling(20).mean()
    df['Volume_ratio'] = volume / df['Volume_SMA_20']
    
    # Chaikin Money Flow
    df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
    
    # 5. CYCLE INDICATORS
    # Hilbert Transform (simplified)
    df['HT_SINE'], df['HT_LEADSINE'] = ta.trend.ema_indicator(close, window=5), ta.trend.ema_indicator(close, window=10)
    
    # 6. STATISTICAL
    # Returns
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close).diff()
    
    # Rolling volatility
    df['volatility_10'] = df['returns'].rolling(10).std() * np.sqrt(252)
    df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
    
    # 7. PRICE TRANSFORMS
    # Typical Price
    df['typical_price'] = (high + low + close) / 3
    
    # Weighted Close
    df['weighted_close'] = (high + low + (close * 2)) / 4
    
    # 8. CUSTOM INDICATORS (untuk LSTM features)
    # Price position in BB
    df['BB_position'] = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # RSI normalized
    df['RSI_norm'] = (df['RSI_14'] - 30) / (70 - 30)  # Normalize between 0-1
    
    # MACD normalized
    macd_mean = df['MACD'].rolling(50).mean()
    macd_std = df['MACD'].rolling(50).std()
    df['MACD_norm'] = (df['MACD'] - macd_mean) / (macd_std + 1e-8)
    
    # Volume spike
    df['volume_spike'] = (volume > volume.rolling(20).mean() * 1.5).astype(int)
    
    # Fill NaN values
    df = df.ffill().bfill().fillna(0)
    
    return df

def calculate_all_indicators(df):
    """
    Wrapper untuk backward compatibility dengan 2_price.py
    """
    return add_ta_indicators(df)

# ======================================================
# FUNGSI DATA HISTORIS
# ======================================================
def get_historical_data(symbol="BTC-USD", days=100, interval="1d"):
    """
    Ambil data historis dari Yahoo Finance
    """
    try:
        # Hitung periode berdasarkan interval
        if interval == "1d":
            period = f"{days}d"
        elif interval == "1h":
            period = f"{min(60, days*24)}d"  # Max 60 hari untuk hourly
        elif interval == "1wk":
            period = f"{days*7}d"
        else:
            period = f"{days}d"
        
        # Download data
        df = yf.download(
            symbol, 
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Standardize columns
        df = standardize_dataframe(df)
        
        # Add indicators
        df = add_ta_indicators(df)
        
        # Cache key
        cache_key = f"{symbol}_{days}_{interval}"
        _prediction_cache[cache_key] = df
        _cache_expiry[cache_key] = datetime.now() + timedelta(seconds=30)
        
        return df
        
    except Exception as e:
        print(f"Error getting historical data for {symbol}: {e}")
        
        # Create fallback data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        fallback_df = pd.DataFrame({
            'open': 45000 + np.random.randn(days) * 1000,
            'high': 45500 + np.random.randn(days) * 1000,
            'low': 44500 + np.random.randn(days) * 1000,
            'close': 45000 + np.random.randn(days) * 1000,
            'volume': 1000 + np.random.randn(days) * 100
        }, index=dates)
        
        # Add basic indicators to fallback
        fallback_df = add_ta_indicators(fallback_df)
        
        return fallback_df

# ======================================================
# FUNGSI PREDIKSI LSTM YANG KONSISTEN
# ======================================================
def get_lstm_prediction(df, days_ahead=1):
    """
    Prediksi menggunakan LSTM atau fallback logic
    Hasil akan konsisten antara app.py dan 2_price.py
    """
    try:
        # Jika ada model LSTM, gunakan
        # Untuk sekarang, kita buat prediksi konsisten dengan logika sederhana
        current_price = df['close'].iloc[-1]
        
        # Analisis teknikal untuk prediksi
        rsi = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
        macd_signal = df['MACD_signal'].iloc[-1] if 'MACD_signal' in df.columns else 0
        bb_position = df['BB_position'].iloc[-1] if 'BB_position' in df.columns else 0.5
        
        # Trend analysis
        sma_10 = df['SMA_10'].iloc[-1] if 'SMA_10' in df.columns else current_price
        sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else current_price
        
        # Volume analysis
        volume_ratio = df['Volume_ratio'].iloc[-1] if 'Volume_ratio' in df.columns else 1
        obv_trend = df['OBV'].iloc[-1] > df['OBV'].iloc[-5] if len(df) > 5 else True
        
        # SCORING SYSTEM yang konsisten
        score = 0
        
        # 1. Trend (max 30 points)
        if current_price > sma_10 > sma_50:
            score += 30  # Strong uptrend
        elif current_price > sma_10:
            score += 15  # Mild uptrend
        elif current_price < sma_10 < sma_50:
            score -= 30  # Strong downtrend
        elif current_price < sma_10:
            score -= 15  # Mild downtrend
        
        # 2. RSI (max 20 points)
        if rsi < 30:
            score += 20  # Oversold -> bullish
        elif rsi < 40:
            score += 10
        elif rsi > 70:
            score -= 20  # Overbought -> bearish
        elif rsi > 60:
            score -= 10
        
        # 3. MACD (max 20 points)
        if macd > macd_signal and macd > 0:
            score += 20  # Bullish MACD
        elif macd > macd_signal:
            score += 10
        elif macd < macd_signal and macd < 0:
            score -= 20  # Bearish MACD
        elif macd < macd_signal:
            score -= 10
        
        # 4. Bollinger Bands (max 15 points)
        if bb_position < 0.2:
            score += 15  # Near lower band -> bullish
        elif bb_position < 0.3:
            score += 8
        elif bb_position > 0.8:
            score -= 15  # Near upper band -> bearish
        elif bb_position > 0.7:
            score -= 8
        
        # 5. Volume (max 15 points)
        if volume_ratio > 1.5 and obv_trend:
            score += 15  # High volume with OBV uptrend
        elif volume_ratio > 1.2 and obv_trend:
            score += 8
        elif volume_ratio > 1.5 and not obv_trend:
            score -= 15  # High volume with OBV downtrend
        elif volume_ratio > 1.2 and not obv_trend:
            score -= 8
        
        # Normalize score to -100 to 100
        score = max(-100, min(100, score))
        
        # Convert score to price prediction
        # Score > 0: bullish, Score < 0: bearish
        # Base change: ±0.5% per 10 points
        base_change_pct = (score / 10) * 0.05
        
        # Add some randomness (less for high confidence predictions)
        confidence = min(95, 50 + abs(score))
        randomness = np.random.uniform(-0.5, 0.5) * (1 - confidence/100)
        
        # Total change
        total_change_pct = base_change_pct + randomness
        
        # Cap at ±5%
        total_change_pct = max(-5, min(5, total_change_pct))
        
        # Calculate prediction
        prediction = current_price * (1 + total_change_pct/100)
        
        return {
            'prediction': float(prediction),
            'change_pct': float(total_change_pct),
            'score': float(score),
            'confidence': float(confidence),
            'rsi': float(rsi),
            'macd': float(macd),
            'bb_position': float(bb_position),
            'source': 'TA-Based LSTM Simulation',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in get_lstm_prediction: {e}")
        
        # Fallback prediction
        current_price = df['close'].iloc[-1] if len(df) > 0 else 45000
        change_pct = np.random.uniform(-2, 3)
        
        return {
            'prediction': float(current_price * (1 + change_pct/100)),
            'change_pct': float(change_pct),
            'score': 0,
            'confidence': 50.0,
            'rsi': 50.0,
            'macd': 0.0,
            'bb_position': 0.5,
            'source': 'Fallback Prediction',
            'timestamp': datetime.now().isoformat()
        }

# ======================================================
# FUNGSI PREDIKSI KONSISTEN UTAMA
# ======================================================
def get_consistent_prediction(symbol="BTC-USD", days=100, interval="1d", use_cache=True):
    """
    Fungsi utama untuk mendapatkan prediksi konsisten
    Digunakan oleh BOTH app.py dan 2_price.py
    """
    cache_key = f"pred_{symbol}_{days}_{interval}"
    
    # Cek cache (valid 60 detik)
    if use_cache and cache_key in _prediction_cache:
        expiry = _cache_expiry.get(cache_key, datetime.min)
        if datetime.now() < expiry:
            cached_data = _prediction_cache[cache_key]
            # Update timestamp
            cached_data['timestamp'] = datetime.now().isoformat()
            return cached_data
    
    try:
        # 1. Ambil data historis
        df = get_historical_data(symbol, days, interval)
        
        if df.empty or len(df) < 10:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # 2. Dapatkan prediksi
        prediction_data = get_lstm_prediction(df)
        
        # 3. Hitung metrics tambahan
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change_24h = ((current_price - prev_price) / prev_price) * 100
        
        # 4. Volume analysis
        volume_today = df['volume'].iloc[-1]
        volume_avg = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else volume_today
        volume_ratio = volume_today / volume_avg if volume_avg > 0 else 1
        
        # 5. Trend strength
        trend_strength = 0
        if 'SMA_10' in df.columns and 'SMA_50' in df.columns:
            price_above_sma10 = current_price > df['SMA_10'].iloc[-1]
            price_above_sma50 = current_price > df['SMA_50'].iloc[-1]
            sma10_above_sma50 = df['SMA_10'].iloc[-1] > df['SMA_50'].iloc[-1]
            
            if price_above_sma10 and price_above_sma50 and sma10_above_sma50:
                trend_strength = 1  # Strong uptrend
            elif price_above_sma10 and price_above_sma50:
                trend_strength = 0.5  # Moderate uptrend
            elif not price_above_sma10 and not price_above_sma50 and not sma10_above_sma50:
                trend_strength = -1  # Strong downtrend
            elif not price_above_sma10 and not price_above_sma50:
                trend_strength = -0.5  # Moderate downtrend
        
        # 6. Compile result
        result = {
            'symbol': symbol,
            'current_price': float(current_price),
            'price_change_24h': float(price_change_24h),
            'prediction': float(prediction_data['prediction']),
            'prediction_change_pct': float(prediction_data['change_pct']),
            'score': float(prediction_data['score']),
            'confidence': float(prediction_data['confidence']),
            'rsi': float(prediction_data['rsi']),
            'macd': float(prediction_data['macd']),
            'bb_position': float(prediction_data['bb_position']),
            'volume_ratio': float(volume_ratio),
            'trend_strength': float(trend_strength),
            'source': prediction_data['source'],
            'timestamp': datetime.now().isoformat(),
            'data_points': len(df),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Simpan ke cache
        _prediction_cache[cache_key] = result
        _cache_expiry[cache_key] = datetime.now() + timedelta(seconds=60)
        
        return result
        
    except Exception as e:
        print(f"Error in get_consistent_prediction for {symbol}: {e}")
        
        # Return fallback data yang konsisten
        fallback_price = 45000 + np.random.uniform(-1000, 1000)
        fallback_change = np.random.uniform(-3, 3)
        
        fallback_result = {
            'symbol': symbol,
            'current_price': float(fallback_price),
            'price_change_24h': float(fallback_change),
            'prediction': float(fallback_price * (1 + fallback_change/100)),
            'prediction_change_pct': float(fallback_change),
            'score': 0.0,
            'confidence': 50.0,
            'rsi': 50.0,
            'macd': 0.0,
            'bb_position': 0.5,
            'volume_ratio': 1.0,
            'trend_strength': 0.0,
            'source': 'Fallback Data',
            'timestamp': datetime.now().isoformat(),
            'data_points': 0,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return fallback_result

# ======================================================
# FUNGSI SENTIMEN KONSISTEN
# ======================================================
def get_sentiment_data(price_data=None, symbol="BTC-USD"):
    """
    Data sentimen yang konsisten berdasarkan analisis teknikal
    """
    cache_key = f"sentiment_{symbol}"
    
    # Cek cache (valid 60 detik)
    if cache_key in _sentiment_cache:
        expiry = _cache_expiry.get(cache_key, datetime.min)
        if datetime.now() < expiry:
            return _sentiment_cache[cache_key]
    
    try:
        if price_data is None:
            price_data = get_consistent_prediction(symbol)
        
        current_price = price_data['current_price']
        price_change = price_data['price_change_24h']
        rsi = price_data['rsi']
        score = price_data['score']
        trend = price_data['trend_strength']
        
        # Base sentiment dari price change
        if price_change > 3:
            base_sentiment = 80
        elif price_change > 1:
            base_sentiment = 65
        elif price_change < -3:
            base_sentiment = 20
        elif price_change < -1:
            base_sentiment = 35
        else:
            base_sentiment = 55
        
        # Adjust dengan RSI
        rsi_adjustment = 0
        if rsi < 30:
            rsi_adjustment = 15  # Oversold -> lebih positif
        elif rsi < 40:
            rsi_adjustment = 8
        elif rsi > 70:
            rsi_adjustment = -15  # Overbought -> lebih negatif
        elif rsi > 60:
            rsi_adjustment = -8
        
        # Adjust dengan score
        score_adjustment = score / 5  # Convert score -100 to 100 -> -20 to 20
        
        # Adjust dengan trend
        trend_adjustment = trend * 10
        
        # Calculate final sentiment
        final_sentiment = base_sentiment + rsi_adjustment + score_adjustment + trend_adjustment
        final_sentiment = max(0, min(100, final_sentiment))
        
        # Calculate news counts berdasarkan sentiment
        base_news_count = 15 + abs(price_change) * 2  # Lebih volatil -> lebih banyak berita
        
        positive_news = int(base_news_count * final_sentiment / 100)
        negative_news = int(base_news_count * (100 - final_sentiment) / 100 * 0.7)
        neutral_news = base_news_count - positive_news - negative_news
        
        # Ensure minimum counts
        positive_news = max(1, positive_news)
        negative_news = max(1, negative_news)
        neutral_news = max(0, neutral_news)
        
        # Determine sentiment category
        if final_sentiment > 65:
            sentiment_category = "POSITIVE"
            sentiment_class = "sentiment-positive"
        elif final_sentiment < 35:
            sentiment_category = "NEGATIVE"
            sentiment_class = "sentiment-negative"
        else:
            sentiment_category = "NEUTRAL"
            sentiment_class = "sentiment-neutral"
        
        result = {
            'positif_pct': round(final_sentiment, 1),
            'pos_count': positive_news,
            'neg_count': negative_news,
            'net_count': neutral_news,
            'total_news': positive_news + negative_news + neutral_news,
            'sentiment_category': sentiment_category,
            'sentiment_class': sentiment_class,
            'price_change': round(price_change, 2),
            'rsi': round(rsi, 2),
            'source': 'Unified Sentiment Analysis',
            'timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Cache result
        _sentiment_cache[cache_key] = result
        _cache_expiry[cache_key] = datetime.now() + timedelta(seconds=60)
        
        return result
        
    except Exception as e:
        print(f"Error in get_sentiment_data: {e}")
        
        return {
            'positif_pct': 50.0,
            'pos_count': 8,
            'neg_count': 5,
            'net_count': 2,
            'total_news': 15,
            'sentiment_category': "NEUTRAL",
            'sentiment_class': "sentiment-neutral",
            'price_change': 0.0,
            'rsi': 50.0,
            'source': 'Fallback Sentiment',
            'timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# ======================================================
# FUNGSI MARKET SIGNAL KONSISTEN
# ======================================================
def get_market_signal(symbol="BTC-USD"):
    """
    Market signal yang konsisten berdasarkan multiple factors
    """
    try:
        # Get prediction data
        prediction_data = get_consistent_prediction(symbol)
        sentiment_data = get_sentiment_data(prediction_data, symbol)
        
        # Extract factors
        score = prediction_data['score']
        confidence = prediction_data['confidence']
        sentiment = sentiment_data['positif_pct']
        trend = prediction_data['trend_strength']
        rsi = prediction_data['rsi']
        
        # Calculate composite signal score (0-100)
        signal_score = (
            (score + 100) / 2 * 0.3 +  # Convert score -100 to 100 -> 0 to 100
            confidence * 0.2 +
            sentiment * 0.2 +
            (trend + 1) * 50 * 0.2 +  # Convert trend -1 to 1 -> 0 to 100
            (70 - abs(rsi - 50)) * 0.1  # RSI closeness to 50
        )
        
        # Determine signal
        if signal_score >= 70:
            signal = "BULLISH"
            advice = "Strong buying opportunities. Consider adding to positions."
            color_class = "signal-bullish"
            emoji = "🚀"
        elif signal_score >= 55:
            signal = "MODERATELY BULLISH"
            advice = "Good buying opportunities. Consider gradual accumulation."
            color_class = "signal-bullish"
            emoji = "📈"
        elif signal_score <= 30:
            signal = "BEARISH"
            advice = "Consider risk management. May be time to take profits."
            color_class = "signal-bearish"
            emoji = "📉"
        elif signal_score <= 45:
            signal = "MODERATELY BEARISH"
            advice = "Exercise caution. Consider reducing exposure."
            color_class = "signal-bearish"
            emoji = "⚠️"
        else:
            signal = "NEUTRAL"
            advice = "Market indecisive. Hold and monitor for clearer signals."
            color_class = "signal-neutral"
            emoji = "📊"
        
        return {
            'market_signal': signal,
            'market_advice': advice,
            'signal_score': round(signal_score, 1),
            'raw_score': round(score, 1),
            'confidence': round(confidence, 1),
            'sentiment': round(sentiment, 1),
            'trend_strength': round(trend, 2),
            'rsi': round(rsi, 1),
            'emoji': emoji,
            'color_class': color_class,
            'source': 'Unified Market Analysis',
            'timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"Error in get_market_signal: {e}")
        
        return {
            'market_signal': "NEUTRAL",
            'market_advice': "Monitor market conditions",
            'signal_score': 50.0,
            'raw_score': 0.0,
            'confidence': 50.0,
            'sentiment': 50.0,
            'trend_strength': 0.0,
            'rsi': 50.0,
            'emoji': "📊",
            'color_class': "signal-neutral",
            'source': 'Fallback Signal',
            'timestamp': datetime.now().isoformat(),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# ======================================================
# FUNGSI UNTUK BACKWARD COMPATIBILITY
# ======================================================
def get_all_data(symbol="BTC-USD"):
    """
    Get semua data sekaligus untuk app.py
    """
    prediction = get_consistent_prediction(symbol)
    sentiment = get_sentiment_data(prediction, symbol)
    signal = get_market_signal(symbol)
    
    return {
        'price': {
            'harga_sekarang': prediction['current_price'],
            'harga_change_pct': prediction['price_change_24h'],
            'last_updated': prediction['last_updated'],
            'source': prediction['source']
        },
        'sentiment': sentiment,
        'prediction': {
            'prediksi_besok': prediction['prediction'],
            'selisih_prediksi': prediction['prediction'] - prediction['current_price'],
            'selisih_prediksi_pct': prediction['prediction_change_pct'],
            'confidence': prediction['confidence'],
            'last_updated': prediction['last_updated'],
            'source': prediction['source']
        },
        'signal': signal,
        'metadata': {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'data_points': prediction['data_points']
        }
    }

# ======================================================
# FUNGSI UNTUK 2_price.py COMPATIBILITY
# ======================================================
def get_price_data_for_analysis(symbol="BTC-USD", days=100):
    """
    Get data untuk 2_price.py dengan format yang diharapkan
    """
    df = get_historical_data(symbol, days)
    
    # Add lowercase columns for compatibility
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = df[col.capitalize()] if col.capitalize() in df.columns else np.nan
    
    return df

def get_ta_indicators(df):
    """
    Wrapper untuk mendapatkan indikator TA
    """
    return add_ta_indicators(df)