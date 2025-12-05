# pages/1_sentimen.py - UPDATE DENGAN FUNGSI SCRAPING LANGSUNG
# Tambahkan root project ke PYTHONPATH
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
utils_path = os.path.join(project_root, "utils")

if utils_path not in sys.path:
    sys.path.append(utils_path)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ===========================
# Import fungsi dari utils
# ===========================
# Tambahkan parent directory ke path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.influx_helper import get_sentiment_history, get_news_raw
    from utils.scraper_sentiment import run_scraping
    SCRAPER_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Import error: {e}")
    SCRAPER_AVAILABLE = False

# ===========================
# Konfigurasi Halaman
# ===========================
st.set_page_config(layout="wide", page_title="Analisis Sentimen Bitcoin")
st.title("📰 Analisis Sentimen Bitcoin")

# ===========================
# Sidebar
# ===========================
with st.sidebar:
    st.header("⚙️ Kontrol Data")
    
    # Rentang waktu
    days_sentiment = st.slider("Rentang Hari (Sentimen):", 1, 30, 7)
    days_news = st.slider("Rentang Hari (Berita):", 1, 30, 3)
    
    st.divider()
    
    # Tombol Scraping
    st.subheader("🔄 Update Data")
    
    if SCRAPER_AVAILABLE:
        if st.button("🚀 Scraping Sekarang", type="primary", use_container_width=True):
            with st.spinner("Sedang scraping berita Bitcoin..."):
                result = run_scraping()
                
                if result["success"]:
                    st.success(f"✅ {result['message']}")
                    
                    # Tampilkan statistik
                    if "stats" in result:
                        stats = result["stats"]
                        st.info(f"""
                        **Statistik:**
                        - Total berita: {stats['total_news']}
                        - Positif: {stats['positive']}
                        - Netral: {stats['neutral']}
                        - Negatif: {stats['negative']}
                        """)
                    
                    # Refresh data setelah 2 detik
                    time.sleep(2)
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
    else:
        st.warning("⚠️ Fungsi scraping tidak tersedia")
        
    # Debug mode
    st.divider()
    debug_mode = st.checkbox("🔍 Mode Debug", value=False)

# ===========================
# Load data dari InfluxDB
# ===========================
@st.cache_data(ttl=300)  # Cache 5 menit
def load_sentiment_data(days=7):
    """Load sentiment data from InfluxDB"""
    try:
        df = get_sentiment_history(days=days)
        if df.empty:
            st.warning("Data sentimen kosong dari InfluxDB")
        return df
    except Exception as e:
        st.error(f"Error loading sentiment data: {str(e)[:100]}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache 5 menit
def load_news_data(days=3):
    """Load news data from InfluxDB"""
    try:
        df = get_news_raw(days=days)
        if df.empty:
            st.warning("Data berita kosong dari InfluxDB")
        return df
    except Exception as e:
        st.error(f"Error loading news data: {str(e)[:100]}")
        return pd.DataFrame()

# Load data
with st.spinner("Memuat data dari InfluxDB..."):
    df = load_sentiment_data(days_sentiment)
    df_news = load_news_data(days_news)

# ===========================
# Debug Information
# ===========================
if debug_mode and not df.empty:
    with st.expander("🔍 Debug Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sentiment Data")
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {list(df.columns)}")
            if not df.empty:
                st.write(f"Data range: {df['_time'].min()} to {df['_time'].max()}")
                st.dataframe(df.head())
        
        with col2:
            st.subheader("📰 News Data")
            st.write(f"Shape: {df_news.shape}")
            st.write(f"Columns: {list(df_news.columns)}")
            if not df_news.empty:
                st.write(f"Data range: {df_news['_time'].min()} to {df_news['_time'].max()}")
                st.dataframe(df_news.head())

# ===========================
# Handle empty data
# ===========================
if df.empty:
    st.warning("⚠️ Tidak ada data sentimen yang ditemukan di InfluxDB.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Solusi:**
        1. Klik tombol **🚀 Scraping Sekarang** di sidebar
        2. Tunggu proses selesai
        3. Data akan otomatis muncul
        """)
    
    with col2:
        if SCRAPER_AVAILABLE:
            if st.button("🔧 Jalankan Scraping Pertama", type="primary", use_container_width=True):
                with st.spinner("Memulai scraping pertama..."):
                    result = run_scraping()
                    if result["success"]:
                        st.success("✅ Scraping berhasil! Refreshing...")
                        time.sleep(3)
                        st.cache_data.clear()
                        st.rerun()
        else:
            st.error("Fungsi scraping tidak tersedia")
    
    st.stop()

# ===========================
# Preprocess data
# ===========================
# Pastikan format datetime
if '_time' in df.columns:
    df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
    df = df.dropna(subset=['_time'])
    df = df.sort_values('_time')
    
    # Buat sentiment score
    if all(col in df.columns for col in ['positive', 'neutral', 'negative']):
        total = df['positive'] + df['neutral'] + df['negative']
        df['score'] = (df['positive'] - df['negative']) / total.replace(0, 1)
    else:
        df['score'] = 0

# ===========================
# Header dengan info data
# ===========================
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    if not df.empty:
        latest_date = df['_time'].max().strftime('%Y-%m-%d')
        st.metric("📅 Data Terakhir", latest_date)
with col_info2:
    st.metric("📊 Total Data", len(df))
with col_info3:
    if not df.empty and 'score' in df.columns:
        avg_score = df['score'].mean()
        st.metric("📈 Rata-rata Score", f"{avg_score:.3f}")

# ===========================
# Layout utama
# ===========================
col1, col2 = st.columns([1, 2])

# ---------- Gauge Sentimen ----------
with col1:
    st.subheader("💓 Sentimen Terkini")
    
    if not df.empty and 'score' in df.columns:
        latest_score = df['score'].iloc[-1]
        latest_date = df['_time'].iloc[-1].strftime('%Y-%m-%d %H:%M')
        
        if latest_score > 0.1:
            sentiment_label = f"Positif 😊 ({latest_date})"
            color = "green"
        elif latest_score < -0.1:
            sentiment_label = f"Negatif 😟 ({latest_date})"
            color = "red"
        else:
            sentiment_label = f"Netral 😐 ({latest_date})"
            color = "yellow"
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': sentiment_label},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': color},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightcoral"},
                    {'range': [-0.3, 0.3], 'color': "lightyellow"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ---------- Tren Sentimen ----------
with col2:
    st.subheader("📈 Tren Sentimen")
    
    if not df.empty and 'score' in df.columns:
        fig_trend = px.line(
            df, x='_time', y='score',
            markers=True,
            title=f'Tren Sentimen ({days_sentiment} hari terakhir)',
            labels={'_time': 'Tanggal', 'score': 'Sentimen Score'}
        )
        fig_trend.update_layout(
            hovermode='x unified',
            xaxis_title="Tanggal",
            yaxis_title="Sentimen Score"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

# ===========================
# Distribusi Sentimen (DIUBAH - berdasarkan rentang waktu)
# ===========================
st.subheader("📊 Distribusi Sentimen")

if all(col in df.columns for col in ['positive', 'neutral', 'negative']):
    col_dist1, col_dist2 = st.columns([2, 1])
    
    with col_dist1:
        # PILIHAN: Total atau Per Hari
        display_mode = st.radio(
            "Mode Distribusi:",
            ["Total dalam Rentang", "Rata-rata per Hari"],
            horizontal=True,
            key="dist_mode"
        )
        
        if display_mode == "Total dalam Rentang":
            # Hitung total untuk seluruh rentang waktu
            total_pos = df['positive'].sum()
            total_neu = df['neutral'].sum()
            total_neg = df['negative'].sum()
            total_all = total_pos + total_neu + total_neg
            title = 'Distribusi Total Sentimen'
            
        else:  # Rata-rata per Hari
            # Hitung rata-rata per hari
            avg_pos = df['positive'].mean()
            avg_neu = df['neutral'].mean()
            avg_neg = df['negative'].mean()
            total_all = avg_pos + avg_neu + avg_neg
            title = 'Distribusi Rata-rata Sentimen per Hari'
        
        # Buat pie chart
        if display_mode == "Total dalam Rentang":
            fig_pie = px.pie(
                values=[total_pos, total_neu, total_neg],
                names=['Positif', 'Netral', 'Negatif'],
                title=title,
                color_discrete_map={'Positif': 'green', 'Netral': 'yellow', 'Negatif': 'red'}
            )
        else:
            fig_pie = px.pie(
                values=[avg_pos, avg_neu, avg_neg],
                names=['Positif', 'Netral', 'Negatif'],
                title=title,
                color_discrete_map={'Positif': 'green', 'Netral': 'yellow', 'Negatif': 'red'}
            )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_dist2:
        st.write(f"**Statistik ({display_mode}):**")
        
        if display_mode == "Total dalam Rentang":
            if total_all > 0:
                st.metric("👍 Positif", f"{int(total_pos)} ({total_pos/total_all*100:.1f}%)")
                st.metric("😐 Netral", f"{int(total_neu)} ({total_neu/total_all*100:.1f}%)")
                st.metric("👎 Negatif", f"{int(total_neg)} ({total_neg/total_all*100:.1f}%)")
                st.caption(f"📅 Selama {days_sentiment} hari")
        else:
            if total_all > 0:
                st.metric("👍 Positif", f"{avg_pos:.1f} ({avg_pos/total_all*100:.1f}%)")
                st.metric("😐 Netral", f"{avg_neu:.1f} ({avg_neu/total_all*100:.1f}%)")
                st.metric("👎 Negatif", f"{avg_neg:.1f} ({avg_neg/total_all*100:.1f}%)")
                st.caption("📊 Rata-rata per hari")

# ===========================
# Tabel Berita Terbaru
# ===========================
st.subheader("📰 Berita Terbaru")

if not df_news.empty:
    # Preprocess
    df_news['_time'] = pd.to_datetime(df_news['_time'], errors='coerce')
    df_news = df_news.sort_values('_time', ascending=False)
    
    # Filter kolom yang ada
    available_cols = []
    for col in ['_time', 'title', 'source', 'sentiment', 'sentiment_score']:
        if col in df_news.columns:
            available_cols.append(col)
    
    if available_cols:
        # Tampilkan info
        st.info(f"📅 **{len(df_news)} berita** dalam {days_news} hari terakhir")
        
        # Tampilkan tabel
        st.dataframe(
            df_news[available_cols].head(15),
            use_container_width=True,
            height=400,
            column_config={
                "_time": st.column_config.DatetimeColumn("Waktu", format="YYYY-MM-DD HH:mm"),
                "title": "Judul",
                "source": "Sumber",
                "sentiment": "Sentimen",
                "sentiment_score": st.column_config.NumberColumn("Score", format="%.3f")
            }
        )
        
        # Tombol lihat semua
        if len(df_news) > 15:
            with st.expander("Lihat Semua Berita"):
                st.dataframe(df_news[available_cols])
else:
    st.info("📭 Belum ada data berita. Coba lakukan scraping atau perbesar rentang waktu.")

# ===========================
# Footer dengan kontrol cepat
# ===========================
st.markdown("---")
col_footer1, col_footer2 = st.columns(2)

with col_footer1:
    st.caption(f"🕐 Terakhir diperbarui: {datetime.now().strftime('%H:%M:%S')}")

with col_footer2:
    if st.button("🔄 Refresh Sekarang", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


