# 🌪️ Hệ Hỗ Trợ Ra Quyết Định Chuỗi Cung Ứng Lai (ML - Fuzzy - MILP)

Đây là kho lưu trữ mã nguồn cho Hệ thống Quản trị Chuỗi cung ứng tích hợp Trí tuệ Nhân tạo và Vận trù học, được phát triển phục vụ cho Luận văn Thạc sĩ. Hệ thống giải quyết bài toán thời gian giao hàng động (Dynamic Lead Time) dưới tác động của rủi ro thời tiết cực đoan theo thời gian thực.

## 🧠 Kiến trúc Hệ thống (4 Lớp)
1. **Machine Learning (Demand Forecast):** Dự báo nhu cầu nguyên vật liệu bằng Random Forest.
2. **AI Risk Assessor (Lớp 1.5):** Quét dữ liệu thời tiết thực tế (Open-Meteo API) để chấm điểm rủi ro bão (Risk Score 0-10).
3. **Fuzzy Logic (Tồn kho An toàn Động):** Hệ chuyên gia nội suy linh hoạt vạch báo động tồn kho dựa trên rủi ro ngoại sinh.
4. **MILP Optimization (Tối ưu Lịch trình):** Quy hoạch nguyên hỗn hợp giải quyết bài toán Dynamic Lead Time, tự động gọi hàng lùi về quá khứ để lách bão, cực tiểu hóa chi phí (Min TC).

## 🚀 Hướng dẫn Cài đặt & Chạy Hệ thống
Hệ thống được thiết kế theo chuẩn End-to-End Pipeline. 

**Cài đặt thư viện:**
```bash
pip install pandas numpy scikit-learn scikit-fuzzy pulp streamlit requests matplotlib
