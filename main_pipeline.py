import os
import subprocess
import time
import sys

def run_script(script_name, command_type="python"):
    """
    Hàm thực thi một tệp Python độc lập, đảm bảo không xung đột bộ nhớ.
    """
    print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ ĐANG KHỞI CHẠY: {script_name}...")
    try:
        if command_type == "python":
            # Dùng sys.executable để trỏ đúng vào môi trường Python hiện tại
            result = subprocess.run([sys.executable, script_name], check=True)
        elif command_type == "streamlit":
            # Chạy Streamlit ngầm và không chặn luồng console
            print(f"[{time.strftime('%H:%M:%S')}] 🌐 Đang khởi động Máy chủ Giao diện Web (Dashboard)...")
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", script_name])
            return True
            
        print(f"[{time.strftime('%H:%M:%S')}] ✅ HOÀN THÀNH: {script_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"\n❌ LỖI NGHIÊM TRỌNG: Quá trình chạy '{script_name}' bị thất bại!")
        print("🛑 Dừng toàn bộ hệ thống. Vui lòng kiểm tra lại lỗi ở trên.")
        return False
    except FileNotFoundError:
        print(f"\n❌ LỖI: Không tìm thấy tệp '{script_name}' trong thư mục!")
        return False

def orchestrate_pipeline():
    """
    Quy trình điều phối (Pipeline) chuẩn CI/CD cho Hệ thống Chuỗi cung ứng Lai
    """
    print("="*65)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG QUẢN TRỊ CHUỖI CUNG ỨNG (END-TO-END PIPELINE)")
    print("="*65)
    
    # BƯỚC 1: KHỞI TẠO DỮ LIỆU THỜI TIẾT (Nếu chưa có)
    if not os.path.exists("Weather_Delivery_History.csv"):
        print("\n[HỆ THỐNG] Không tìm thấy Dữ liệu Thời tiết. Tiến hành tạo mới giả lập...")
        if not run_script("generate_weather_data.py"): return
    else:
        print("\n[HỆ THỐNG] ✅ Dữ liệu Lịch sử Thời tiết đã sẵn sàng.")

    # BƯỚC 2: HUẤN LUYỆN LỚP 1.5 - AI RỦI RO THỜI TIẾT
    if not os.path.exists("weather_risk_model.pkl"):
        print("\n[HỆ THỐNG] Não bộ AI Thời tiết chưa được huấn luyện. Đang tiến hành học...")
        if not run_script("layer1_5_weather_risk.py"): return
    else:
        print("[HỆ THỐNG] ✅ Mô hình AI Dự báo Rủi ro đã sẵn sàng.")

    # BƯỚC 3: CHẠY LỚP 1 - AI DỰ BÁO NHU CẦU TIÊU THỤ THAN
    print("\n[HỆ THỐNG] Đang chạy Lớp 1 (Machine Learning) để dự báo tiêu thụ 7 ngày tới...")
    if not run_script("layer1_machinelearning.py"): return

    # BƯỚC 4: KHỞI ĐỘNG DASHBOARD (TÍCH HỢP LỚP 2 & LỚP 3 BÊN TRONG)
    print("\n" + "="*65)
    print("🎉 TẤT CẢ MODULE CHẠY NGẦM ĐÃ XONG! ĐANG MỞ BẢNG ĐIỀU KHIỂN...")
    print("="*65)
    
    time.sleep(2) # Nghỉ 2 giây để hệ thống ghi file hoàn tất
    run_script("app_dashboard.py", command_type="streamlit")

if __name__ == "__main__":
    orchestrate_pipeline()