import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_weather_delivery_data(filename="Weather_Delivery_History.csv", days=1095):
    """
    Sinh dữ liệu lịch sử Thời tiết và Giao hàng trong 3 năm (1095 ngày)
    để huấn luyện mô hình Machine Learning Lớp 1.5.
    """
    print("⏳ Đang khởi tạo dữ liệu Thời tiết & Giao hàng giả lập (3 năm)...")
    
    np.random.seed(42) # Cố định seed để dữ liệu đồng nhất mỗi lần sinh
    
    # Tạo chuỗi thời gian quá khứ (từ 3 năm trước đến hiện tại)
    end_date = datetime(2026, 9, 7) # Lấy mốc thời gian hiện tại
    start_date = end_date - timedelta(days=days)
    date_list = [start_date + timedelta(days=x) for x in range(days)]
    
    # 1. Sinh đặc trưng thời tiết ngẫu nhiên (Features)
    # Nhiệt độ: 20 - 38 độ C
    temperature = np.random.normal(loc=28, scale=4, size=days).clip(15, 40)
    
    # Lượng mưa: 70% số ngày không mưa, 30% có mưa (từ nhỏ đến bão)
    is_raining = np.random.choice([0, 1], size=days, p=[0.7, 0.3])
    rainfall_mm = is_raining * np.random.exponential(scale=30, size=days).clip(0, 300)
    
    # Sức gió: Trung bình 15 km/h, những ngày mưa to gió sẽ mạnh hơn
    wind_speed_kmh = np.random.normal(loc=15, scale=5, size=days) + (rainfall_mm * 0.2)
    
    # Cảnh báo bão: Mưa > 100mm VÀ Gió > 50km/h
    storm_warning = ((rainfall_mm > 100) & (wind_speed_kmh > 50)).astype(int)
    
    # 2. Xây dựng logic Trễ giao hàng và Điểm rủi ro (Target Variables)
    delay_days = np.zeros(days)
    risk_score = np.zeros(days)
    
    for i in range(days):
        base_delay = 0
        base_risk = np.random.uniform(0.5, 2.5) # Rủi ro cơ bản luôn tồn tại
        
        # Thêm rủi ro do Mưa
        if rainfall_mm[i] > 30:
            base_delay += 1
            base_risk += np.random.uniform(2, 4)
        if rainfall_mm[i] > 100:
            base_delay += 1
            base_risk += np.random.uniform(2, 3)
            
        # Thêm rủi ro do Gió
        if wind_speed_kmh[i] > 40:
            base_delay += 1
            base_risk += np.random.uniform(1, 3)
            
        # Rủi ro do Bão (Trễ nghiêm trọng)
        if storm_warning[i] == 1:
            base_delay += np.random.choice([1, 2])
            base_risk = np.random.uniform(8.5, 10.0)
            
        # Chốt số liệu (Delay tối đa 5 ngày, Rủi ro giới hạn 0-10)
        delay_days[i] = min(5, base_delay)
        risk_score[i] = min(10.0, base_risk)
        
        # Nếu giao đúng hạn, rủi ro ép về mức thấp
        if delay_days[i] == 0:
            risk_score[i] = min(3.0, risk_score[i])
            
    # 3. Đóng gói vào DataFrame
    df = pd.DataFrame({
        'Ngay': date_list,
        'Nhiet_Do_C': np.round(temperature, 1),
        'Luong_Mua_mm': np.round(rainfall_mm, 1),
        'Suc_Gio_kmh': np.round(wind_speed_kmh, 1),
        'Bao': storm_warning,
        'So_Ngay_Tre': delay_days.astype(int),
        'Diem_Rui_Ro': np.round(risk_score, 1)
    })
    
    # Lưu ra file CSV
    df.to_csv(filename, index=False)
    
    print("✅ Đã tạo thành công file: 'Weather_Delivery_History.csv'")
    print(f"📊 Tổng quan dữ liệu ({days} ngày):")
    print(f"   - Số ngày bão / thời tiết cực đoan: {df['Bao'].sum()} ngày")
    print(f"   - Số ngày giao hàng bị trễ: {(df['So_Ngay_Tre'] > 0).sum()} ngày")
    print("\nTrích xuất 5 dòng mẫu:")
    print(df.sample(5).to_string(index=False))

if __name__ == "__main__":
    generate_synthetic_weather_delivery_data()