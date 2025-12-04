# utils/forecasting_utils.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def forecast_future_prices(df, model, scaler, meta, days_ahead=7):
    """
    Fungsi utama forecasting - tetap pertahankan yang ini
    """
    try:
        # Import fungsi build_features_for_model
        from utils.feature_builder import build_features_for_model
        
        # 1. Ambil parameter dari metadata
        features_used = meta.get('features_used', [])
        window_size = meta.get('best_parameters', {}).get('window_size', 
                        meta.get('timesteps', 10))
        
        print(f"🔮 Forecasting {days_ahead} hari ke depan")
        print(f"   Window size: {window_size}")
        print(f"   Features used: {len(features_used)}")
        
        # 2. Bangun fitur untuk data historis
        df_features = build_features_for_model(df.copy(), meta)
        
        if df_features.empty:
            raise ValueError("Tidak dapat membangun fitur untuk forecasting")
        
        # 3. Scale data historis
        scaled_data = scaler.transform(df_features)
        
        # 4. Ambil sequence terakhir untuk forecasting
        last_sequence = scaled_data[-window_size:]
        
        # 5. Buat forecasting secara rekursif
        forecast_scaled = []
        current_sequence = last_sequence.copy()
        
        # Cari index fitur 'close' dalam features_used
        close_idx = None
        for i, feat in enumerate(features_used):
            if isinstance(feat, str) and 'close' in feat.lower():
                close_idx = i
                print(f"✅ 'close' column found at index {i}: {feat}")
                break
        
        if close_idx is None:
            close_idx = 3  # Default index
        
        for day in range(days_ahead):
            # Reshape untuk model
            input_seq = current_sequence.reshape(1, window_size, len(features_used))
            
            # Prediksi satu langkah ke depan
            pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
            forecast_scaled.append(pred_scaled)
            
            # Update sequence: geser window dan tambahkan prediksi
            new_row = current_sequence[-1].copy()  # Ambil baris terakhir
            
            # Update nilai close dengan prediksi
            new_row[close_idx] = pred_scaled
            
            # Geser window: hapus baris pertama, tambahkan baris baru
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # 6. Inverse transform prediksi
        dummy_forecast = np.zeros((len(forecast_scaled), len(features_used)))
        dummy_forecast[:, close_idx] = forecast_scaled
        
        # Inverse transform
        forecast_inversed = scaler.inverse_transform(dummy_forecast)
        
        # Ambil kolom close
        forecast_prices = forecast_inversed[:, close_idx]
        
        # 7. **FIX INI: Cara yang benar membuat tanggal forecast**
        last_date = df.index[-1]
        
        # Pastikan last_date adalah Timestamp
        if not isinstance(last_date, pd.Timestamp):
            last_date = pd.to_datetime(last_date)
        
        # **CARA YANG BENAR: Gunakan pd.DateOffset**
        forecast_dates = []
        for i in range(1, days_ahead + 1):
            # Tambahkan hari dengan DateOffset (cara yang benar di pandas baru)
            next_date = last_date + pd.DateOffset(days=i)
            forecast_dates.append(next_date)
        
        forecast_dates = pd.DatetimeIndex(forecast_dates)
        
        # Buat DataFrame dengan tanggal yang benar
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast_Price': forecast_prices,
            'Symbol': 'Unknown'
        })
        
        print(f"✅ Forecasting selesai! Prediksi {days_ahead} hari ke depan")
        print(f"   Rentang harga: ${forecast_prices.min():.2f} - ${forecast_prices.max():.2f}")
        print(f"   Tanggal forecast: {forecast_dates[0]} hingga {forecast_dates[-1]}")
        
        return forecast_df, forecast_prices
        
    except Exception as e:
        print(f"❌ Error in forecast_future_prices: {e}")
        import traceback
        traceback.print_exc()
        raise

def forecast_future_prices_simple(df, model, scaler, meta, days_ahead=7):
    """
    Versi sederhana dari forecasting untuk menghindari masalah datetime
    """
    try:
        # Import fungsi build_features_for_model
        from utils.feature_builder import build_features_for_model
        
        # 1. Ambil parameter dari metadata
        features_used = meta.get('features_used', [])
        window_size = meta.get('best_parameters', {}).get('window_size', 10)
        
        print(f"🔮 [SIMPLE] Forecasting {days_ahead} hari ke depan")
        
        # 2. Bangun fitur untuk data historis
        df_features = build_features_for_model(df.copy(), meta)
        
        if df_features.empty:
            raise ValueError("Tidak dapat membangun fitur untuk forecasting")
        
        # 3. Scale data historis
        scaled_data = scaler.transform(df_features)
        
        # 4. Ambil sequence terakhir untuk forecasting
        last_sequence = scaled_data[-window_size:]
        
        # 5. Buat forecasting secara rekursif
        forecast_scaled = []
        current_sequence = last_sequence.copy()
        
        # Cari index fitur 'close' dalam features_used
        close_idx = 3  # Default index
        for i, feat in enumerate(features_used):
            if isinstance(feat, str) and 'close' in feat.lower():
                close_idx = i
                print(f"✅ [SIMPLE] 'close' column found at index {i}: {feat}")
                break
        
        for day in range(days_ahead):
            # Reshape untuk model
            input_seq = current_sequence.reshape(1, window_size, len(features_used))
            
            # Prediksi satu langkah ke depan
            pred_scaled = model.predict(input_seq, verbose=0)[0, 0]
            forecast_scaled.append(pred_scaled)
            
            # Update sequence
            new_row = current_sequence[-1].copy()
            new_row[close_idx] = pred_scaled
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # 6. Inverse transform prediksi
        dummy_forecast = np.zeros((len(forecast_scaled), len(features_used)))
        dummy_forecast[:, close_idx] = forecast_scaled
        forecast_inversed = scaler.inverse_transform(dummy_forecast)
        forecast_prices = forecast_inversed[:, close_idx]
        
        # 7. **SOLUSI SIMPLE: Gunakan tanggal relatif jika ada masalah**
        last_date = df.index[-1]
        
        # Coba buat tanggal dengan cara yang aman
        try:
            # Pastikan last_date adalah Timestamp
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.to_datetime(last_date)
            
            # Buat tanggal dengan DateOffset
            forecast_dates = []
            for i in range(1, days_ahead + 1):
                next_date = last_date + pd.DateOffset(days=i)
                forecast_dates.append(next_date)
            
            forecast_dates = pd.DatetimeIndex(forecast_dates)
            
        except Exception as date_error:
            print(f"⚠️ [SIMPLE] Masalah dengan tanggal: {date_error}")
            print("⚠️ [SIMPLE] Menggunakan tanggal dari sekarang sebagai fallback")
            
            # Fallback: buat tanggal dari sekarang
            from datetime import datetime, timedelta
            today = datetime.now()
            forecast_dates = [today + timedelta(days=i) for i in range(1, days_ahead + 1)]
            forecast_dates = pd.to_datetime(forecast_dates)
        
        # Buat DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast_Price': forecast_prices
        })
        
        print(f"✅ [SIMPLE] Forecasting selesai! Prediksi {days_ahead} hari ke depan")
        print(f"   Rentang harga: ${forecast_prices.min():.2f} - ${forecast_prices.max():.2f}")
        
        return forecast_df, forecast_prices
        
    except Exception as e:
        print(f"❌ Error in forecast_future_prices_simple: {e}")
        import traceback
        traceback.print_exc()
        raise

def calculate_forecast_metrics(forecast_prices, current_price=None):
    """
    Menghitung metrik untuk hasil forecasting
    """
    metrics = {}
    
    if len(forecast_prices) > 0:
        # Jika hanya ada 1 prediksi
        if len(forecast_prices) == 1:
            if current_price is not None and current_price > 0:
                metrics['total_change_pct'] = ((forecast_prices[0] - current_price) / current_price) * 100
            else:
                metrics['total_change_pct'] = 0
            metrics['min_price'] = forecast_prices[0]
            metrics['max_price'] = forecast_prices[0]
            metrics['trend'] = 'UP' if metrics.get('total_change_pct', 0) > 0 else 'DOWN' if metrics.get('total_change_pct', 0) < 0 else 'FLAT'
        else:
            # Daily returns
            returns = []
            for i in range(1, len(forecast_prices)):
                if forecast_prices[i-1] > 0:
                    ret = ((forecast_prices[i] - forecast_prices[i-1]) / forecast_prices[i-1]) * 100
                    returns.append(ret)
            
            if returns:
                metrics['avg_daily_change'] = np.mean(returns)
                metrics['volatility'] = np.std(returns)
            else:
                metrics['avg_daily_change'] = 0
                metrics['volatility'] = 0
            
            if forecast_prices[0] > 0:
                metrics['total_change_pct'] = ((forecast_prices[-1] - forecast_prices[0]) / forecast_prices[0]) * 100
            else:
                metrics['total_change_pct'] = 0
            
            metrics['min_price'] = np.min(forecast_prices)
            metrics['max_price'] = np.max(forecast_prices)
            metrics['trend'] = 'UP' if metrics.get('total_change_pct', 0) > 0 else 'DOWN' if metrics.get('total_change_pct', 0) < 0 else 'FLAT'
        
        # Confidence levels
        if len(forecast_prices) > 0:
            last_price = forecast_prices[-1]
            metrics['confidence_high'] = last_price * 1.05  # +5%
            metrics['confidence_low'] = last_price * 0.95   # -5%
    
    return metrics