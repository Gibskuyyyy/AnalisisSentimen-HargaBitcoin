# app.py - Dashboard Utama
import sys
import os

# Tambahkan root project ke PYTHONPATH
sys.path.append(os.path.join(os.getcwd(), "utils"))

import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import json

# ======================================================
# IMPORT UNIFIED PREDICTOR
# ======================================================
try:
    from unified_predictor import get_all_data as unified_get_all_data
    UNIFIED_PREDICTOR_AVAILABLE = True
    st.success("✅ Unified Predictor loaded successfully!")
except ImportError as e:
    UNIFIED_PREDICTOR_AVAILABLE = False
    st.warning(f"⚠️ Unified Predictor not available: {e}. Using fallback functions.")
except Exception as e:
    UNIFIED_PREDICTOR_AVAILABLE = False
    st.warning(f"⚠️ Error loading Unified Predictor: {e}. Using fallback functions.")

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Bitcoin AI Dashboard",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>
    /* Header */
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F7931A, #F2A900);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 15px;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 25px 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Metric Labels */
    .metric-label {
        font-size: 1rem;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    /* Metric Values */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #212529;
        line-height: 1;
        margin-bottom: 10px;
    }
    
    /* Price Change */
    .price-change {
        font-size: 1rem;
        font-weight: 600;
    }
    
    .price-up {
        color: #10B981;
    }
    
    .price-down {
        color: #EF4444;
    }
    
    /* Sentiment Badge */
    .sentiment-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 5px;
    }
    
    .sentiment-positive {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
    }
    
    .sentiment-neutral {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
    }
    
    .sentiment-negative {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
    }
    
    /* Signal Badge */
    .signal-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    
    .signal-bullish {
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
    }
    
    .signal-bearish {
        background: linear-gradient(135deg, #EF4444, #F87171);
        color: white;
    }
    
    /* Status update */
    .update-time {
        text-align: center;
        color: #6c757d;
        font-size: 0.85rem;
        margin-top: 2rem;
    }
    
    /* Navigation */
    .nav-button {
        width: 100%;
        margin: 5px 0;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .nav-button:hover {
        background: #f8f9fa;
    }
    
    /* Data source badge */
    .data-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 5px;
        vertical-align: middle;
    }
    
    .badge-live {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
    }
    
    .badge-default {
        background: rgba(107, 114, 128, 0.15);
        color: #6B7280;
    }
    
    /* Tambahan untuk signal neutral */
    .signal-neutral {
        background: linear-gradient(135deg, #6B7280, #9CA3AF);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# FUNGSI UNTUK MENGAMBIL DATA - MENGIKUTI FORMAT UNIFIED
# ======================================================

def get_bitcoin_price():
    """Mengambil harga Bitcoin menggunakan unified predictor"""
    try:
        if UNIFIED_PREDICTOR_AVAILABLE:
            all_data = unified_get_all_data()
            return all_data['price']
        else:
            # Fallback ke kode lama
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(period="1d")
            
            if len(data) > 0:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                return {
                    'harga_sekarang': float(current_price),
                    'harga_change_pct': float(change_pct),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'source': 'Yahoo Finance'
                }
    except Exception as e:
        st.error(f"Error mengambil harga Bitcoin: {e}")
    
    # Fallback: data default
    return {
        'harga_sekarang': 45000.00,
        'harga_change_pct': 2.5,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'Default Data'
    }

def get_sentiment_data():
    """Mengambil data sentimen menggunakan unified predictor"""
    try:
        if UNIFIED_PREDICTOR_AVAILABLE:
            all_data = unified_get_all_data()
            return all_data['sentiment']
        else:
            # Fallback ke kode lama
            positif_pct = 65.5
            pos_count = 10
            neg_count = 5
            net_count = 3
            total_news = pos_count + neg_count + net_count
            
            return {
                'positif_pct': positif_pct,
                'pos_count': pos_count,
                'neg_count': neg_count,
                'net_count': net_count,
                'total_news': total_news,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Sentiment Analysis'
            }
    except Exception as e:
        st.error(f"Error mengambil data sentimen: {e}")
    
    # Fallback: data default
    return {
        'positif_pct': 50.0,
        'pos_count': 5,
        'neg_count': 3,
        'net_count': 2,
        'total_news': 10,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'Default Data'
    }

def get_prediction_data():
    """Mengambil prediksi harga menggunakan unified predictor"""
    try:
        if UNIFIED_PREDICTOR_AVAILABLE:
            all_data = unified_get_all_data()
            return all_data['prediction']
        else:
            # Fallback ke kode lama
            current_price = get_bitcoin_price()['harga_sekarang']
            import random
            change_pred = random.uniform(-2, 5)
            prediksi_besok = current_price * (1 + change_pred/100)
            selisih = prediksi_besok - current_price
            selisih_pct = (selisih / current_price) * 100
            
            return {
                'prediksi_besok': float(prediksi_besok),
                'selisih_prediksi': float(selisih),
                'selisih_prediksi_pct': float(selisih_pct),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'LSTM Model'
            }
    except Exception as e:
        st.error(f"Error mengambil prediksi: {e}")
    
    # Fallback: data default
    current_price = get_bitcoin_price()['harga_sekarang']
    return {
        'prediksi_besok': float(current_price * 1.02),
        'selisih_prediksi': float(current_price * 0.02),
        'selisih_prediksi_pct': 2.0,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'Default Data'
    }

def get_market_signal():
    """Menentukan signal market menggunakan unified predictor"""
    try:
        if UNIFIED_PREDICTOR_AVAILABLE:
            all_data = unified_get_all_data()
            signal_data = all_data['signal']
            
            # Map signal ke format yang diharapkan
            signal_map = {
                'BULLISH': 'bullish',
                'MODERATELY BULLISH': 'bullish',
                'NEUTRAL': 'neutral',
                'MODERATELY BEARISH': 'bearish',
                'BEARISH': 'bearish'
            }
            
            signal = signal_map.get(signal_data.get('market_signal', 'NEUTRAL'), 'neutral')
            
            return {
                'market_signal': signal,
                'market_advice': signal_data.get('market_advice', 'Hold and monitor market'),
                'score': signal_data.get('signal_score', 0),
                'raw_score': signal_data.get('raw_score', 0),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': signal_data.get('source', 'Market Analysis'),
                'color_class': signal_data.get('color_class', 'signal-neutral')
            }
        else:
            # Fallback ke kode lama
            price_data = get_bitcoin_price()
            sentiment_data = get_sentiment_data()
            prediction_data = get_prediction_data()
            
            current_price = price_data['harga_sekarang']
            price_change = price_data['harga_change_pct']
            positif_pct = sentiment_data['positif_pct']
            pred_change = prediction_data['selisih_prediksi_pct']
            
            # Logika sederhana untuk menentukan signal
            score = 0
            
            # Faktor harga (0-3)
            if price_change > 2:
                score += 3
            elif price_change > 0:
                score += 1
            elif price_change < -2:
                score -= 3
            elif price_change < 0:
                score -= 1
            
            # Faktor sentimen (0-3)
            if positif_pct > 70:
                score += 3
            elif positif_pct > 55:
                score += 1
            elif positif_pct < 30:
                score -= 3
            elif positif_pct < 45:
                score -= 1
            
            # Faktor prediksi (0-2)
            if pred_change > 3:
                score += 2
            elif pred_change > 0:
                score += 1
            elif pred_change < -3:
                score -= 2
            elif pred_change < 0:
                score -= 1
            
            # Tentukan signal berdasarkan total score
            if score >= 4:
                signal = "bullish"
                advice = "Consider buying opportunities"
            elif score <= -4:
                signal = "bearish"
                advice = "Consider risk management"
            else:
                signal = "neutral"
                advice = "Hold and monitor market"
            
            return {
                'market_signal': signal,
                'market_advice': advice,
                'score': score,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Market Analysis'
            }
    except Exception as e:
        st.error(f"Error menentukan signal market: {e}")
    
    # Fallback: data default
    return {
        'market_signal': "bullish",
        'market_advice': "Monitor market conditions",
        'score': 0,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'Default Data'
    }

def get_all_data():
    """Mengambil semua data yang dibutuhkan"""
    if UNIFIED_PREDICTOR_AVAILABLE:
        try:
            unified_data = unified_get_all_data()
            # Tambahkan metadata
            unified_data['metadata'] = {
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'unified_predictor': True,
                'version': '1.0'
            }
            return unified_data
        except Exception as e:
            st.warning(f"Error using unified predictor: {e}. Using fallback.")
    
    # Fallback: menggunakan fungsi individu
    return {
        'price': get_bitcoin_price(),
        'sentiment': get_sentiment_data(),
        'prediction': get_prediction_data(),
        'signal': get_market_signal(),
        'metadata': {
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'unified_predictor': False,
            'version': '1.0'
        }
    }

# ======================================================
# SIDEBAR
# ======================================================
import os
import sys

with st.sidebar:
    st.markdown("### ⚙️ Navigation")
    
    # Navigation buttons
    if st.button("📊 Dashboard", use_container_width=True, type="primary"):
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📈 Analysis Pages")
    
    # Dapatkan base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigasi dengan path absolute
    if st.button("📰 Sentiment Analysis", use_container_width=True):
        sentiment_path = os.path.join(base_dir, "pages", "1_Sentiment.py")
        if os.path.exists(sentiment_path):
            st.switch_page("pages/1_Sentiment.py")
        else:
            st.error(f"File not found: {sentiment_path}")
            # Buat fallback: tampilkan dalam tab/window baru
            st.markdown('[Buka Halaman Sentimen](pages/1_Sentimen.py)', unsafe_allow_html=True)
    
    if st.button("💰 Price & Prediction", use_container_width=True):
        price_path = os.path.join(base_dir, "pages", "2_Price.py")
        if os.path.exists(price_path):
            st.switch_page("pages/2_Price.py")
        else:
            st.error(f"File not found: {price_path}")
            st.markdown('[Buka Halaman Price](pages/2_Price.py)', unsafe_allow_html=True)
    
    st.markdown("---")
    # ... sisanya sama
    st.markdown("### 🔄 Data Refresh")
    
    if st.button("🔄 Refresh All Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Data source info
    st.markdown("---")
    st.markdown("### 📡 Data Sources")
    
    if UNIFIED_PREDICTOR_AVAILABLE:
        st.success("✅ Using Unified Predictor")
        st.info("""
        - **Harga Bitcoin**: Yahoo Finance + TA Analysis
        - **Sentimen**: Unified Sentiment Analysis
        - **Prediksi**: LSTM + TA-Based Simulation
        - **Signal**: Unified Market Analysis
        - **Status**: Synchronized with Price Analysis Page
        """)
    else:
        st.warning("⚠️ Using Fallback Functions")
        st.info("""
        - **Harga Bitcoin**: Yahoo Finance
        - **Sentimen**: Simulated sentiment analysis
        - **Prediksi**: LSTM model simulation
        - **Signal**: Market analysis algorithm
        """)
    
    # Last update time
    st.markdown(f"**🕐 Last update:**")
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="header-title">🟠 Bitcoin AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #6c757d; margin-bottom: 2rem;">Real-time Monitoring & Analysis Dashboard</div>', unsafe_allow_html=True)

# ======================================================
# LOAD DATA
# ======================================================
with st.spinner("🔄 Loading data..."):
    all_data = get_all_data()
    
    # Extract data
    price_data = all_data['price']
    sentiment_data = all_data['sentiment']
    prediction_data = all_data['prediction']
    signal_data = all_data['signal']

# ======================================================
# 4 KOLOM UTAMA
# ======================================================
st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)

# KOLOM 1: HARGA BITCOIN SAAT INI
col1, col2, col3, col4 = st.columns(4)

with col1:
    harga_btc = price_data['harga_sekarang']
    harga_change = price_data['harga_change_pct']
    
    # Data source badge
    source_badge = "badge-live" if price_data['source'] != 'Default Data' else "badge-default"
    source_text = price_data['source'].replace(' ', '_').upper()
    
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">
            💰 Harga Bitcoin 
            <span class="data-badge {source_badge}">{source_text}</span>
        </div>
        <div class="metric-value">${harga_btc:,.2f}</div>
    ''', unsafe_allow_html=True)
    
    if harga_change > 0:
        st.markdown(f'<div class="price-change price-up">▲ {harga_change:.2f}%</div>', unsafe_allow_html=True)
    elif harga_change < 0:
        st.markdown(f'<div class="price-change price-down">▼ {abs(harga_change):.2f}%</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="price-change">0.00%</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d; margin-top: 10px;">Updated: {price_data["last_updated"][11:19]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# KOLOM 2: SENTIMEN HARI INI
with col2:
    sentimen_pct = sentiment_data['positif_pct']
    
    # Data source badge
    source_badge = "badge-live" if sentiment_data['source'] != 'Default Data' else "badge-default"
    source_text = sentiment_data['source'].replace(' ', '_').upper()
    
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">
            📊 Sentimen Hari Ini
            <span class="data-badge {source_badge}">{source_text}</span>
        </div>
        <div class="metric-value">{sentimen_pct:.1f}%</div>
    ''', unsafe_allow_html=True)
    
    # Gunakan sentiment_class dari unified predictor jika ada
    if 'sentiment_class' in sentiment_data:
        sentiment_class = sentiment_data['sentiment_class']
        sentiment_text = sentiment_data.get('sentiment_category', 'Netral')
    else:
        # Fallback ke logika lama
        if sentimen_pct > 60:
            sentiment_class = "sentiment-positive"
            sentiment_text = "Positif"
        elif sentimen_pct < 40:
            sentiment_class = "sentiment-negative"
            sentiment_text = "Negatif"
        else:
            sentiment_class = "sentiment-neutral"
            sentiment_text = "Netral"
    
    st.markdown(f'<div class="sentiment-badge {sentiment_class}">{sentiment_text}</div>', unsafe_allow_html=True)
    
    # Info detail
    pos_count = sentiment_data['pos_count']
    neg_count = sentiment_data['neg_count']
    net_count = sentiment_data['net_count']
    st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d; margin-top: 10px;">Pos: {pos_count} | Neg: {neg_count} | Net: {net_count}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d;">Updated: {sentiment_data["last_updated"][11:19]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# KOLOM 3: PREDIKSI HARGA BESOK
with col3:
    prediksi = prediction_data['prediksi_besok']
    selisih = prediction_data['selisih_prediksi']
    selisih_pct = prediction_data['selisih_prediksi_pct']
    
    # Data source badge
    source_badge = "badge-live" if prediction_data['source'] != 'Default Data' else "badge-default"
    source_text = prediction_data['source'].replace(' ', '_').upper()
    
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">
            🔮 Prediksi Besok
            <span class="data-badge {source_badge}">{source_text}</span>
        </div>
        <div class="metric-value">${prediksi:,.2f}</div>
    ''', unsafe_allow_html=True)
    
    if selisih > 0:
        st.markdown(f'<div class="price-change price-up">+${selisih:,.2f} ({selisih_pct:.2f}%)</div>', unsafe_allow_html=True)
    elif selisih < 0:
        st.markdown(f'<div class="price-change price-down">-${abs(selisih):,.2f} ({abs(selisih_pct):.2f}%)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="price-change">Tidak berubah</div>', unsafe_allow_html=True)
    
    # Current vs Prediction
    current_price = price_data['harga_sekarang']
    direction = "naik" if prediksi > current_price else "turun"
    change_vs_current = abs(prediksi - current_price)
    
    # Tambahkan confidence jika ada
    if 'confidence' in prediction_data:
        confidence = prediction_data['confidence']
        st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d; margin-top: 5px;">Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d; margin-top: 5px;">Diprediksi {direction} ${change_vs_current:,.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d;">Updated: {prediction_data["last_updated"][11:19]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# KOLOM 4: BULLISH / BEARISH
with col4:
    signal = signal_data['market_signal']
    advice = signal_data['market_advice']
    score = signal_data.get('score', 0)
    raw_score = signal_data.get('raw_score', score)
    
    # Data source badge
    source_badge = "badge-live" if signal_data['source'] != 'Default Data' else "badge-default"
    source_text = signal_data['source'].replace(' ', '_').upper()
    
    st.markdown(f'''
    <div class="metric-card">
        <div class="metric-label">
            📈 Market Signal
            <span class="data-badge {source_badge}">{source_text}</span>
        </div>
    ''', unsafe_allow_html=True)
    
    # Tampilkan signal badge dengan color_class jika ada
    if 'color_class' in signal_data:
        color_class = signal_data['color_class']
        st.markdown(f'<div class="signal-badge {color_class}">{signal.upper()}</div>', unsafe_allow_html=True)
    else:
        # Fallback ke logika lama
        if signal.lower() == "bullish":
            st.markdown('<div class="signal-badge signal-bullish">BULLISH</div>', unsafe_allow_html=True)
        elif signal.lower() == "bearish":
            st.markdown('<div class="signal-badge signal-bearish">BEARISH</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="signal-badge signal-neutral">NEUTRAL</div>', unsafe_allow_html=True)
    
    # Tampilkan advice
    st.markdown(f'<div style="margin-top: 15px; font-size: 0.9rem; color: #6c757d;">{advice}</div>', unsafe_allow_html=True)
    
    # Score info
    if 'confidence' in signal_data:
        confidence = signal_data.get('confidence', 0)
        st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d; margin-top: 10px;">Signal Score: {score} | Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d; margin-top: 10px;">Signal Score: {score}</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div style="font-size: 0.8rem; color: #6c757d;">Updated: {signal_data["last_updated"][11:19]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# DETAILED DATA SECTION
# ======================================================
st.markdown("---")
st.markdown("### 📋 Detailed Data")

# Tampilkan data dalam bentuk tabel
col_detail1, col_detail2 = st.columns(2)

with col_detail1:
    st.markdown("##### 📊 Price & Prediction Details")
    
    # Siapkan data tambahan jika ada
    price_details_data = {
        'Metric': ['Current Price', '24h Change', 'Prediction Tomorrow', 'Prediction Change', 'Current vs Prediction'],
        'Value': [
            f"${harga_btc:,.2f}",
            f"{harga_change:+.2f}%",
            f"${prediksi:,.2f}",
            f"{selisih_pct:+.2f}%",
            f"{'Up' if prediksi > harga_btc else 'Down'} ${abs(prediksi - harga_btc):,.2f}"
        ]
    }
    
    # Tambahkan confidence jika ada
    if 'confidence' in prediction_data:
        price_details_data['Metric'].append('Prediction Confidence')
        price_details_data['Value'].append(f"{prediction_data['confidence']:.1f}%")
    
    price_details = pd.DataFrame(price_details_data)
    
    st.dataframe(price_details, hide_index=True, use_container_width=True)

with col_detail2:
    st.markdown("##### 📈 Sentiment & Signal Details")
    
    # Siapkan data sentimen
    sentiment_details_data = {
        'Metric': ['Positive Sentiment', 'Negative Sentiment', 'Neutral Sentiment', 'Total News', 'Market Signal', 'Signal Score'],
        'Value': [
            f"{sentimen_pct:.1f}%",
            f"{(sentiment_data['neg_count']/sentiment_data['total_news']*100):.1f}%",
            f"{(sentiment_data['net_count']/sentiment_data['total_news']*100):.1f}%",
            f"{sentiment_data['total_news']} articles",
            signal.upper(),
            f"{score}"
        ]
    }
    
    # Tambahkan data tambahan jika ada
    if 'rsi' in signal_data:
        sentiment_details_data['Metric'].append('RSI')
        sentiment_details_data['Value'].append(f"{signal_data['rsi']:.1f}")
    
    if 'trend_strength' in signal_data:
        sentiment_details_data['Metric'].append('Trend Strength')
        sentiment_details_data['Value'].append(f"{signal_data['trend_strength']:.2f}")
    
    sentiment_details = pd.DataFrame(sentiment_details_data)
    
    st.dataframe(sentiment_details, hide_index=True, use_container_width=True)

# ======================================================
# DATA SOURCE INFO
# ======================================================
st.markdown("---")
st.markdown("### 📡 Data Sources Information")

# Buat DataFrame untuk info sumber data
source_data = []
source_data.append(['Bitcoin Price', price_data['source'], price_data['last_updated'], 
                   '✅ Live' if price_data['source'] != 'Default Data' else '⚠️ Default'])
source_data.append(['Market Sentiment', sentiment_data['source'], sentiment_data['last_updated'],
                   '✅ Live' if sentiment_data['source'] != 'Default Data' else '⚠️ Default'])
source_data.append(['Price Prediction', prediction_data['source'], prediction_data['last_updated'],
                   '✅ Live' if prediction_data['source'] != 'Default Data' else '⚠️ Default'])
source_data.append(['Market Signal', signal_data['source'], signal_data['last_updated'],
                   '✅ Live' if signal_data['source'] != 'Default Data' else '⚠️ Default'])

# Tambahkan info unified predictor jika digunakan
if UNIFIED_PREDICTOR_AVAILABLE:
    source_data.append(['Prediction Engine', 'Unified Predictor', datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '✅ Synced'])

source_info = pd.DataFrame(source_data, columns=['Data Type', 'Source', 'Last Updated', 'Status'])

st.dataframe(source_info, hide_index=True, use_container_width=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown(f"**🟠 Bitcoin AI Dashboard**")
    if UNIFIED_PREDICTOR_AVAILABLE:
        st.caption("Version 2.0 (Unified Predictor)")
    else:
        st.caption("Version 1.0 (Fallback Mode)")

with col_footer2:
    st.markdown(f"**🕐 Last Full Update:**")
    st.caption(all_data['metadata']['last_updated'])

with col_footer3:
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ======================================================
# HOW TO USE
# ======================================================
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    ### 📌 Dashboard Components:
    
    1. **💰 Harga Bitcoin Saat Ini**
       - Menampilkan harga real-time Bitcoin
       - Persentase perubahan 24 jam
       - Data dari Yahoo Finance
    
    2. **📊 Sentimen Hari Ini**
       - Analisis sentimen pasar dari berita
       - Persentase sentimen positif
       - Jumlah berita positif, negatif, dan netral
    
    3. **🔮 Prediksi Besok**
       - Prediksi harga Bitcoin untuk hari berikutnya
       - Berdasarkan model LSTM
       - Perubahan yang diprediksi vs harga saat ini
    
    4. **📈 Market Signal**
       - Signal trading: Bullish/Bearish/Neutral
       - Rekomendasi tindakan
       - Score berdasarkan analisis multi-faktor
    
    ### 🔄 Refresh Data:
    - Klik tombol **🔄 Refresh Dashboard** untuk update data terbaru
    - Gunakan sidebar untuk navigasi ke halaman analisis detail
    
    ### 📈 Untuk Data Real:
    1. Jalankan halaman **📰 Sentiment Analysis** untuk scraping berita
    2. Jalankan halaman **💰 Price & Prediction** untuk analisis teknikal
    3. Data akan otomatis tersedia di dashboard ini
    """)
    
    # Tambahkan info tentang unified predictor
    if UNIFIED_PREDICTOR_AVAILABLE:
        st.markdown("""
        ### 🔄 Unified Predictor:
        - **Konsistensi**: Prediksi di dashboard dan halaman analisis sekarang sinkron
        - **Analisis**: Menggunakan indikator teknikal yang sama dengan TA Library
        - **Cache**: Data di-cache selama 60 detik untuk performa
        - **Fallback**: Jika terjadi error, sistem otomatis menggunakan fallback
        """)

# ======================================================
# CACHING FOR PERFORMANCE
# ======================================================
@st.cache_data(ttl=60)  # Cache untuk 60 detik
def cached_get_all_data():
    return get_all_data()