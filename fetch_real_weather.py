import requests
import pandas as pd
import numpy as np
from datetime import datetime

def get_real_weather_7_days(lat=20.95, lon=107.08):
    """
    Kết nối API Thời tiết Open-Meteo (Miễn phí, Không cần API Key)
    Mặc định đang lấy tọa độ của Cảng biển Quảng Ninh (Vịnh Hạ Long)
    """
    print(f"📡 Đang kết nối vệ tinh thời tiết tại tọa độ (Lat: {lat}, Lon: {lon})...")
    
    # URL API của Open-Meteo yêu cầu lấy dự báo 7 ngày tới
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"daily=temperature_2m_max,rain_sum,windspeed_10m_max&"
        f"timezone=Asia%2FHo_Chi_Minh"
    )
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Trích xuất dữ liệu từ API
        daily_data = data['daily']
        dates = daily_data['time']
        temps = daily_data['temperature_2m_max']
        rains = daily_data['rain_sum']
        winds = daily_data['windspeed_10m_max']
        
        # Chuyển đổi thành DataFrame khớp với Lớp 1.5 của bạn
        weather_df = pd.DataFrame({
            'Ngày': dates,
            'Nhiệt Độ (C)': np.round(temps, 1),
            'Lượng Mưa (mm)': np.round(rains, 1),
            'Sức Gió (km/h)': np.round(winds, 1)
        })
        
        # Tự động tính toán biến 'Có Bão' dựa trên logic: Mưa > 100mm VÀ Gió > 50km/h
        weather_df['Có Bão (1/0)'] = ((weather_df['Lượng Mưa (mm)'] > 100) & (weather_df['Sức Gió (km/h)'] > 50)).astype(int)
        
        print("✅ Đã tải thành công Dữ liệu Thời tiết Thực tế 7 Ngày tới!")
        print(weather_df.to_string(index=False))
        
        return weather_df

    except Exception as e:
        print(f"❌ Lỗi kết nối API: {e}")
        return None

if __name__ == "__main__":
    get_real_weather_7_days()