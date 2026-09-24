import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings

warnings.filterwarnings('ignore')

def train_weather_risk_model(data_path="Weather_Delivery_History.csv", model_path="weather_risk_model.pkl"):
    """
    Huấn luyện AI học cách đánh giá mức độ rủi ro chuỗi cung ứng dựa trên thời tiết.
    """
    print("🧠 [1/3] Đang tải dữ liệu Lịch sử Thời tiết & Giao hàng...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{data_path}'. Hãy chạy Bước 1 trước.")
        return None

    # Xác định Biến độc lập (X) và Biến phụ thuộc (y)
    features = ['Nhiet_Do_C', 'Luong_Mua_mm', 'Suc_Gio_kmh', 'Bao']
    target = 'Diem_Rui_Ro'

    X = df[features]
    y = df[target]

    print("⏳ [2/3] Đang phân chia dữ liệu và Huấn luyện Random Forest (Lớp 1.5)...")
    # Tách 80% để Học, 20% để Thi (Test) - Chuẩn học thuật
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Khởi tạo mô hình Rừng ngẫu nhiên với 100 cây quyết định
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    # Học xong thì phải làm bài kiểm tra (Đánh giá trên tập X_test)
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "="*55)
    print(" BÁO CÁO ĐÁNH GIÁ MÔ HÌNH LỚP 1.5 (WEATHER RISK AI)")
    print("="*55)
    print(f" Sai số tuyệt đối trung bình (MAE): {mae:.2f} điểm (Thang 10)")
    print(f" Sai số bình phương trung bình (RMSE): {rmse:.2f} điểm")
    print("="*55 + "\n")

    # Lưu lại "Não bộ" của mô hình xuống ổ cứng
    joblib.dump(rf_model, model_path)
    print(f"✅ [3/3] Đã lưu mô hình trí tuệ nhân tạo thành công tại: '{model_path}'")

    return rf_model

def simulate_predict_7_days(model_path="weather_risk_model.pkl"):
    """
    Hàm mô phỏng API Dự báo thời tiết thực tế cho 7 ngày tới.
    (Giả định có một cơn bão đổ bộ vào Ngày 4 và Ngày 5).
    """
    print("\n🌦️ MÔ PHỎNG KẾT QUẢ DỰ BÁO: RỦI RO CHO 7 NGÀY TỚI:")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy mô hình. Hãy huấn luyện trước.")
        return

    # Kịch bản kịch tính: Ngày 1-3 đẹp trời, Ngày 4-5 bão lớn, Ngày 6-7 bình thường lại
    future_weather = pd.DataFrame({
        'Ngay_Tuong_Lai': [f'Ngày {i}' for i in range(1, 8)],
        'Nhiet_Do_C': [30, 29, 28, 25, 24, 27, 29],
        'Luong_Mua_mm': [0, 10, 30, 150, 200, 20, 0],
        'Suc_Gio_kmh': [12, 15, 25, 70, 85, 30, 10],
        'Bao': [0, 0, 0, 1, 1, 0, 0] # 1 là có bão
    })

    # Dùng mô hình để dự đoán rủi ro
    X_future = future_weather[['Nhiet_Do_C', 'Luong_Mua_mm', 'Suc_Gio_kmh', 'Bao']]
    predicted_risks = model.predict(X_future)

    future_weather['Diem_Rui_Ro_AI_Du_Bao'] = np.round(predicted_risks, 1)

    print(future_weather.to_string(index=False))
    print("\n=> 🚀 Mảng 'Diem_Rui_Ro_AI_Du_Bao' này đã sẵn sàng để tự động nạp vào Lớp 2 (Fuzzy Logic)!")

if __name__ == "__main__":
    print("★" * 60)
    print(" KHỞI ĐỘNG LỚP 1.5: MACHINE LEARNING DỰ BÁO RỦI RO GIAO HÀNG")
    print("★" * 60 + "\n")
    
    # 1. Huấn luyện và lưu mô hình
    train_weather_risk_model()
    
    # 2. Chạy thử nghiệm dự báo
    simulate_predict_7_days()