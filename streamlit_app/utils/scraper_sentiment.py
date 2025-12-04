# utils/scraper_sentiment.py - Scraping ke InfluxDB
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from gnews import GNews
from googletrans import Translator
from textblob import TextBlob
import nltk
import re
import joblib
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from dotenv import load_dotenv

nltk.download('punkt', quiet=True)
load_dotenv()

# ============================
# Config
# ============================
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = "sentiment_daily"
CRYPTO_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")

# ============================
# Load Model
# ============================
try:
    VECTORIZER = joblib.load("models/tfidf_vectorizer.pkl")
    SVM_MODEL = joblib.load("models/svm_crypto_sentiment.pkl")
except:
    print("⚠️ Model tidak ditemukan, gunakan TextBlob saja")
    VECTORIZER = None
    SVM_MODEL = None

translator = Translator()

# ============================
# Preprocessing
# ============================
def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_sentiment(text):
    """Predict sentiment dengan fallback ke TextBlob"""
    if VECTORIZER and SVM_MODEL:
        try:
            clean = clean_text(text)
            vec = VECTORIZER.transform([clean])
            pred = SVM_MODEL.predict(vec)[0]
            score = float(TextBlob(text).sentiment.polarity)
            return pred, score
        except:
            pass
    
    # Fallback ke TextBlob
    try:
        score = float(TextBlob(text).sentiment.polarity)
        if score > 0.1:
            label = "positive"
        elif score < -0.1:
            label = "negative"
        else:
            label = "neutral"
        return label, score
    except:
        return "neutral", 0.0

# ============================
# Fetch News
# ============================
def fetch_gnews():
    """Fetch berita dari GNews"""
    try:
        google_news = GNews(language="id", country="ID", period="1d")
        news_id = google_news.get_news("bitcoin")
        
        google_news_en = GNews(language="en", country="US", period="1d")
        news_en = google_news_en.get_news("bitcoin")
        
        articles = []
        for article in news_id + news_en:
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "date": article.get("published date", ""),
                "source": "gnews",
                "url": article.get("url", "")
            })
        
        df = pd.DataFrame(articles)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
        return df
    except Exception as e:
        print(f"Error fetch GNews: {e}")
        return pd.DataFrame()

def fetch_cryptopanic():
    """Fetch berita dari CryptoPanic"""
    if not CRYPTO_API_KEY:
        print("⚠️ CRYPTO_PANIC_API_KEY tidak ditemukan")
        return pd.DataFrame()
    
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_API_KEY}&kind=news&currencies=BTC"
        r = requests.get(url, timeout=10)
        
        if r.status_code != 200:
            print(f"Error CryptoPanic: {r.status_code}")
            return pd.DataFrame()
        
        data = r.json().get("results", [])
        items = []
        for item in data:
            items.append({
                "title": item.get("title", ""),
                "description": "",
                "date": item.get("published_at", "")[:19] if item.get("published_at") else "",
                "source": "cryptopanic",
                "url": item.get("url", "")
            })
        
        df = pd.DataFrame(items)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
        return df
    except Exception as e:
        print(f"Error fetch CryptoPanic: {e}")
        return pd.DataFrame()

# ============================
# Check Existing Data
# ============================
def check_existing_date(client, date_str, measurement="bitcoin_sentiment"):
    """Cek apakah data sudah ada di InfluxDB"""
    try:
        query = f'''
            from(bucket: "{INFLUX_BUCKET}")
                |> range(start: -3d)
                |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                |> filter(fn: (r) => r["date"] == "{date_str}")
                |> limit(n: 1)
        '''
        tables = client.query_api().query(query, org=INFLUX_ORG)
        return len(tables) > 0
    except:
        return False

# ============================
# Write to InfluxDB
# ============================
def write_sentiment_to_influx(df):
    """Tulis data agregat sentimen ke InfluxDB"""
    if df.empty:
        print("⚠️ Tidak ada data sentimen untuk ditulis")
        return
    
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Group by date
        df['date'] = pd.to_datetime(df['date']).dt.date
        df_group = df.groupby("date").agg(
            positive=("sentiment_label", lambda x: (x == "positive").sum()),
            negative=("sentiment_label", lambda x: (x == "negative").sum()),
            neutral=("sentiment_label", lambda x: (x == "neutral").sum()),
            avg_score=("sentiment_score", "mean"),
            total=("sentiment_label", "count")
        ).reset_index()
        
        for _, row in df_group.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            
            # Skip jika sudah ada
            if check_existing_date(client, date_str, "bitcoin_sentiment"):
                print(f"⏭️ Data {date_str} sudah ada, skip...")
                continue
            
            # Write to bitcoin_sentiment
            point = Point("bitcoin_sentiment") \
                .tag("date", date_str) \
                .field("positive", int(row["positive"])) \
                .field("negative", int(row["negative"])) \
                .field("neutral", int(row["neutral"])) \
                .field("total", int(row["total"])) \
                .field("avg_score", float(row["avg_score"])) \
                .time(datetime.utcnow(), WritePrecision.NS)
            
            write_api.write(bucket=INFLUX_BUCKET, record=point)
            print(f"✅ Data sentimen {date_str} ditulis")
        
        print(f"📊 Total {len(df_group)} hari data sentimen disimpan")

def save_raw_news(df):
    """Simpan berita mentah ke InfluxDB"""
    if df.empty:
        print("⚠️ Tidak ada berita untuk disimpan")
        return
    
    with InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        count = 0
        for _, r in df.iterrows():
            raw_text = f"{r.get('title','')} {r.get('description','')}"
            clean = clean_text(raw_text)[:500]
            sentiment = r.get("sentiment_label", "neutral")
            score = r.get("sentiment_score", 0.0)
            
            point = Point("news_raw") \
                .tag("source", str(r.get("source", "unknown"))) \
                .tag("asset", "BTC") \
                .field("title", str(r.get("title", ""))[:200]) \
                .field("content", str(r.get("description", ""))[:300]) \
                .field("clean_text", clean) \
                .field("sentiment", sentiment) \
                .field("sentiment_score", float(score)) \
                .field("url", str(r.get("url", ""))[:200]) \
                .time(datetime.utcnow(), WritePrecision.NS)
            
            write_api.write(bucket=INFLUX_BUCKET, record=point)
            count += 1
        
        print(f"📰 {count} berita mentah disimpan")

# ============================
# Main Function
# ============================
def run_scraping():
    """Fungsi utama untuk scraping dan menyimpan ke InfluxDB"""
    print("="*60)
    print("🚀 SCRAPING BITCOIN NEWS & SENTIMENT")
    print("="*60)
    
    # 1. Fetch berita
    print("\n📥 Mengambil berita dari GNews...")
    df_gnews = fetch_gnews()
    print(f"   ✓ {len(df_gnews)} berita dari GNews")
    
    print("📥 Mengambil berita dari CryptoPanic...")
    df_crypto = fetch_cryptopanic()
    print(f"   ✓ {len(df_crypto)} berita dari CryptoPanic")
    
    # Gabungkan data
    df = pd.concat([df_gnews, df_crypto], ignore_index=True)
    
    if df.empty:
        print("❌ Tidak ada berita yang ditemukan")
        return {"success": False, "message": "No news found"}
    
    print(f"\n📊 Total {len(df)} berita ditemukan")
    
    # 2. Translate (jika ada bahasa Indonesia)
    print("\n🌍 Menerjemahkan teks...")
    df["translated"] = df["title"].apply(
        lambda x: translator.translate(x, dest="en").text if x else ""
    )
    
    # 3. Analisis sentimen
    print("🤖 Menganalisis sentimen...")
    df[["sentiment_label", "sentiment_score"]] = df["translated"].apply(
        lambda x: pd.Series(predict_sentiment(x))
    )
    
    # Tampilkan statistik
    pos = sum(df['sentiment_label'] == 'positive')
    neu = sum(df['sentiment_label'] == 'neutral')
    neg = sum(df['sentiment_label'] == 'negative')
    
    print(f"\n📈 Distribusi Sentimen:")
    print(f"   👍 Positif: {pos} ({pos/len(df)*100:.1f}%)")
    print(f"   😐 Netral:  {neu} ({neu/len(df)*100:.1f}%)")
    print(f"   👎 Negatif: {neg} ({neg/len(df)*100:.1f}%)")
    
    # 4. Simpan ke InfluxDB
    print("\n💾 Menyimpan ke InfluxDB...")
    try:
        save_raw_news(df)
        write_sentiment_to_influx(df)
        print("✅ Data berhasil disimpan ke InfluxDB!")
    except Exception as e:
        print(f"❌ Error menyimpan ke InfluxDB: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}
    
    print("="*60)
    print("✨ SCRAPING SELESAI")
    print("="*60)
    
    return {
        "success": True,
        "message": f"Scraping berhasil! {len(df)} berita diproses",
        "stats": {
            "total_news": len(df),
            "positive": pos,
            "neutral": neu,
            "negative": neg
        }
    }

# ============================
# Untuk dijalankan sebagai script
# ============================
if __name__ == "__main__":
    run_scraping()