import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import requests
from datetime import datetime, timedelta

# Import các hàm lõi từ Lớp 2 và Lớp 3
from layer2_fuzzy_logic import build_fuzzy_system_thesis
from layer3_optimization import solve_milp_single_material

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & ĐỒNG BỘ THỜI GIAN THỰC
# ==========================================
st.set_page_config(page_title="AI Supply Chain DSS", layout="wide")
st.title("🌪️ HỆ THỐNG QUẢN TRỊ CHUỖI CUNG ỨNG LAI (ML-FUZZY-MILP)")
st.markdown("*Tích hợp AI Dự báo Rủi ro Thời tiết & Tối ưu hóa Thời gian chờ động*")
st.markdown("---")

# TỰ ĐỘNG SINH 7 NGÀY THỰC TẾ (Bắt đầu từ hôm nay)
today = datetime.now()
dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]

# Tải Dữ liệu Lớp 1 (Giữ nguyên toàn vẹn dữ liệu gốc)
try:
    df_ml = pd.read_csv("D_t_forecast_7days.csv")
    ml_dates = df_ml['Ngay'].tolist() # Ngày gốc từ lịch sử ML
    dt_than, dt_cd, dt_cb = df_ml['Dùng_Than'].tolist(), df_ml['Dùng_C.Đốt'].tolist(), df_ml['Dùng_Carbon'].tolist()
except Exception:
    st.error("❌ Lỗi: Không tìm thấy 'D_t_forecast_7days.csv'.")
    st.stop()

try:
    weather_ai = joblib.load("weather_risk_model.pkl")
except Exception:
    st.error("❌ Lỗi: Không tìm thấy 'weather_risk_model.pkl'.")
    st.stop()

# ==========================================
# 2.5. ĐIỀU ĐỘ NĂNG SUẤT DÂY CHUYỀN
# ==========================================
st.sidebar.header("🏭 ĐIỀU ĐỘ SẢN XUẤT")
st.sidebar.info("Kéo thanh trượt để phân bổ linh hoạt công suất. (Mặc định hệ thống AI dự báo ở mốc Tổng 100%)")

# Thay nút gạt (Toggle) bằng Thanh trượt (Slider) linh hoạt từ 0% đến 150%
cap_line1 = st.sidebar.slider("🟢 Công suất Dây chuyền 1 (%)", min_value=0, max_value=150, value=60, step=10)
cap_line2 = st.sidebar.slider("🟢 Công suất Dây chuyền 2 (%)", min_value=0, max_value=150, value=40, step=10)

# Hiển thị tổng công suất để kiểm soát
total_cap = cap_line1 + cap_line2
st.sidebar.markdown(f"**🔥 Tổng công suất thực tế:** `{total_cap}%`")

# Hàm tính toán lại Nhu cầu thực tế dựa trên Tỷ lệ Slider
def calculate_actual_demand(dt_list, c1, c2):
    # Tính tổng hệ số (Ví dụ: 100% + 100% = 2.0)
    total_ratio = (c1 + c2) / 100.0
    return [d * total_ratio for d in dt_list]

# Áp dụng nội suy tiêu thụ mới
dt_than_actual = calculate_actual_demand(dt_than, cap_line1, cap_line2)
dt_cd_actual = calculate_actual_demand(dt_cd, cap_line1, cap_line2)
dt_cb_actual = calculate_actual_demand(dt_cb, cap_line1, cap_line2)

# ==========================================
# 2. HÀM GỌI API THỜI TIẾT THỰC TẾ
# ==========================================
def fetch_api_weather():
    url = "https://api.open-meteo.com/v1/forecast?latitude=20.95&longitude=107.08&daily=temperature_2m_max,rain_sum,windspeed_10m_max&timezone=Asia%2FHo_Chi_Minh"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()['daily']
        
        api_dates = data['time'] # ĐÂY LÀ THỜI GIAN THỰC TẾ (REAL-TIME) TỪ VỆ TINH
        
        df = pd.DataFrame({
            'Ngày': api_dates, 
            'Nhiệt Độ (C)': np.round(data['temperature_2m_max'], 1),
            'Lượng Mưa (mm)': np.round(data['rain_sum'], 1),
            'Sức Gió (km/h)': np.round(data['windspeed_10m_max'], 1)
        })
        df['Có Bão (1/0)'] = ((df['Lượng Mưa (mm)'] > 100) & (df['Sức Gió (km/h)'] > 50)).astype(int)
        return df
    except Exception as e:
        st.error(f"Lỗi API: {e}")
        return None

# Khởi tạo mặc định nếu chưa bấm nút (Dùng tạm ngày của ML)
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = pd.DataFrame({
        'Ngày': ml_dates,
        'Nhiệt Độ (C)': [30.0, 29.5, 28.0, 25.0, 24.0, 27.0, 29.0],
        'Lượng Mưa (mm)': [0.0, 10.0, 30.0, 180.0, 200.0, 20.0, 0.0],
        'Sức Gió (km/h)': [12.0, 15.0, 25.0, 85.0, 90.0, 30.0, 10.0],
        'Có Bão (1/0)': [0, 0, 0, 1, 1, 0, 0]
    })

# ==========================================
# 3. CỘT BÊN TRÁI: BẢNG ĐIỀU KHIỂN LOGISTICS
# ==========================================
st.sidebar.header("⚙️ THÔNG SỐ LOGISTICS & KHO BÃI")

with st.sidebar.expander("⚫ 1. KHO THAN", expanded=True):
    i0_than = st.number_input("Tồn kho Than (I_0):", min_value=0, max_value=50000, value=3000, step=500)
    # THÊM MỚI: Lựa chọn Phương tiện vận tải và Thời gian trễ
    transport_than = st.selectbox("Phương tiện vận tải (Than):", ["Xà lan (Đường thủy)", "Xe tải (Đường bộ)"])
    lt_than = 3 if "Xà lan" in transport_than else 1
    st.caption(f"Thời gian chờ gốc (Lead time): **{lt_than} ngày**")
    
    co_than = st.slider("Cước xe Than (Triệu):", min_value=5, max_value=50, value=15) * 1000000
    ch_than, imax_than = 25000, 50000

with st.sidebar.expander("🔥 2. KHO CHẤT ĐỐT", expanded=False):
    i0_cd = st.number_input("Tồn kho C.Đốt (I_0):", min_value=0, max_value=20000, value=1000, step=200)
    lt_cd = st.slider("Thời gian chờ (Lead Time) - Ngày:", min_value=0, max_value=5, value=1)
    co_cd = st.slider("Cước xe C.Đốt (Triệu):", min_value=2, max_value=30, value=8) * 1000000
    ch_cd, imax_cd = 15000, 20000

with st.sidebar.expander("🪨 3. KHO CARBON", expanded=False):
    i0_cb = st.number_input("Tồn kho Carbon (I_0):", min_value=0, max_value=5000, value=100, step=50)
    lt_cb = st.slider("Thời gian chờ (Lead Time) C.B - Ngày:", min_value=0, max_value=5, value=0)
    co_cb = st.slider("Cước xe Carbon (Triệu):", min_value=1, max_value=20, value=5) * 1000000
    ch_cb, imax_cb = 40000, 5000

# ==========================================
# 4. GIAO DIỆN CHÍNH: LỚP 1.5 - AI THỜI TIẾT
# ==========================================
st.subheader("🌤️ 1. Dữ liệu Thời tiết & AI Dự báo Rủi ro")

col_btn, col_info = st.columns([1, 4])
with col_btn:
    if st.button("📡 Quét Thời Tiết Thực Tế (Hạ Long)", use_container_width=True):
        with st.spinner("Đang kết nối vệ tinh..."):
            real_df = fetch_api_weather()
            if real_df is not None:
                st.session_state.weather_data = real_df 
                st.success("Tải dữ liệu thực tế thành công!")
with col_info:
    st.info("💡 Điểm rủi ro cao sẽ tự động **Cộng thêm số ngày trễ (Delay)** vào Thời gian giao hàng (Lead Time) của bạn.")

edited_weather = st.data_editor(st.session_state.weather_data, num_rows="fixed", use_container_width=True)

X_predict = pd.DataFrame({
    'Nhiet_Do_C': edited_weather['Nhiệt Độ (C)'],
    'Luong_Mua_mm': edited_weather['Lượng Mưa (mm)'],
    'Suc_Gio_kmh': edited_weather['Sức Gió (km/h)'],
    'Bao': edited_weather['Có Bão (1/0)']
})

dynamic_risk_scores = weather_ai.predict(X_predict)

cols = st.columns(7)
for i, col in enumerate(cols):
    risk_val = dynamic_risk_scores[i]
    color = "#d63031" if risk_val >= 7 else "#fdcb6e" if risk_val >= 4 else "#00b894"
    col.markdown(f"<div style='text-align: center; padding: 10px; background-color: #f1f2f6; border-radius: 5px; border-bottom: 5px solid {color};'>"
                 f"<b>Ngày {i+1}</b><br><h3 style='margin:0; color:{color}'>{risk_val:.1f}</h3></div>", 
                 unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 5. CHẠY LỚP 2 & LỚP 3 VỚI RỦI RO VÀ LEAD TIME ĐỘNG
# ==========================================
fuzzy_sim = build_fuzzy_system_thesis(visualize=False)

def calculate_dynamic_isafe(dt_list, max_cap):
    isafe_list = []
    for dt, risk in zip(dt_list, dynamic_risk_scores):
        if dt == 0:
            isafe_list.append(0)
        else:
            demand_pct = min(100, max(0, (dt / max_cap) * 100))
            fuzzy_sim.input['Demand'] = demand_pct
            fuzzy_sim.input['Risk'] = min(10.0, risk)
            fuzzy_sim.compute()
            isafe_list.append(dt * fuzzy_sim.output['Safety_Days'])
    return isafe_list

# Tính an toàn động (Lớp 2) - Nạp nhu cầu thực tế
isafe_than = calculate_dynamic_isafe(dt_than_actual, 1000.0)
isafe_cd = calculate_dynamic_isafe(dt_cd_actual, 500.0)
isafe_cb = calculate_dynamic_isafe(dt_cb_actual, 200.0)

# Chạy Tối ưu MILP (Lớp 3) - Nạp nhu cầu thực tế
res_than = solve_milp_single_material("Than", dt_than_actual, isafe_than, i0_than, co_than, ch_than, imax_than, base_lead_time=lt_than, risk_scores=dynamic_risk_scores)
res_cd = solve_milp_single_material("ChatDot", dt_cd_actual, isafe_cd, i0_cd, co_cd, ch_cd, imax_cd, base_lead_time=lt_cd, risk_scores=dynamic_risk_scores)
res_cb = solve_milp_single_material("Carbon", dt_cb_actual, isafe_cb, i0_cb, co_cb, ch_cb, imax_cb, base_lead_time=lt_cb, risk_scores=dynamic_risk_scores)

# ==========================================
# 6. HIỂN THỊ KẾT QUẢ QUYẾT ĐỊNH (CẬP NHẬT 7 BIẾN)
# ==========================================
total_system_cost = sum([res[1] for res in [res_than, res_cd, res_cb] if res[0]])
st.subheader(f"🏆 2. KẾT QUẢ TỐI ƯU HÓA (ĐÃ TÍNH TRỄ HÀNG): {total_system_cost:,.0f} VNĐ")

# Nếu tổng chi phí > 1 Tỷ, cảnh báo đứt gãy chuỗi cung ứng
if total_system_cost >= 1e9:
    st.error("🚨 **BÁO ĐỘNG ĐỎ:** Hệ thống không thể nhập đủ hàng. Tồn đầu kỳ quá thấp so với thời gian chờ (Lead Time) hoặc có Siêu Bão làm đứt gãy chuỗi cung ứng. Nhà máy sẽ bị thiếu nguyên liệu!")

tab1, tab2, tab3 = st.tabs(["⚫ Kịch bản: THAN", "🔥 Kịch bản: CHẤT ĐỐT", "🪨 Kịch bản: CARBON"])

# Lấy trục thời gian hiển thị ĐỘNG từ bảng thời tiết (API quyết định ngày nào)
current_display_dates = st.session_state.weather_data['Ngày'].tolist()

# ... (Khối code tính toán MILP giữ nguyên) ...

# Truyền thêm current_display_dates vào hàm render_tab
def render_tab(tab, name, res_data, dt, isafe, risk_arr, display_dates):
    with tab:
        success, cost, orders, Q_vals, I_vals, R_vals, L_vals = res_data
        if not success:
            st.error(f"⚠️ Hệ thống MILP của {name} bị sập (Infeasible).")
            return
            
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Tổng Chi Phí", f"{(cost % 1e9):,.0f} đ" if cost < 1e9 else "RẤT LỚN (Phạt thiếu hàng)")
        c2.metric("Số Lần Gọi Điện", f"{orders:,.0f} chuyến")
        c3.metric("Số Lần Hàng Tới Bến", f"{sum(1 for r in R_vals if r > 0)} chuyến")
        
        # Bảng dữ liệu dùng thời gian thực
        df_plot = pd.DataFrame({
            'Ngày': display_dates, 
            'Thời Tiết (Rủi ro)': np.round(risk_arr, 1),
            'Chờ Hàng ($L_t$)': [f"{int(l)} ngày" for l in L_vals],
            'Lệnh GỌI MUA ($Q_t$)': Q_vals,
            'Hàng CẬP BẾN ($R_t$)': R_vals,
            'Tồn Thực Tế ($I_t$)': I_vals
        })
        st.dataframe(df_plot.style.format("{:.1f}", subset=['Thời Tiết (Rủi ro)', 'Lệnh GỌI MUA ($Q_t$)', 'Hàng CẬP BẾN ($R_t$)', 'Tồn Thực Tế ($I_t$)']), use_container_width=True)
        
        # Biểu đồ Vận hành Kho dùng thời gian thực
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(display_dates, I_vals, marker='o', color='blue', linewidth=2, label='Tồn kho thực tế ($I_t$)')
        ax.plot(display_dates, isafe, linestyle='--', color='red', linewidth=2, label='Đường an toàn ($I_{safe}$)')
        
        ax2 = ax.twinx()
        width = 0.35
        x = np.arange(len(display_dates))
        
        ax2.bar(x - width/2, Q_vals, width, alpha=0.6, color='orange', label='📞 Lệnh Đặt Mua ($Q_t$)')
        ax2.bar(x + width/2, R_vals, width, alpha=0.6, color='green', label='🚢 Tàu Cập Bến ($R_t$)')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_dates)
        ax.set_ylabel("Khối lượng Tồn (Tấn)")
        ax2.set_ylabel("Khối lượng Giao Dịch (Tấn)")
        ax.set_title(f"MÔ PHỎNG LỊCH TRÌNH ĐẶT HÀNG TRƯỚC (DYNAMIC LEAD TIME) - {name.upper()}", fontweight='bold')
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        st.pyplot(fig)

# Gọi hàm với tham số mới
render_tab(tab1, "Than", res_than, dt_than_actual, isafe_than, dynamic_risk_scores, current_display_dates)
render_tab(tab2, "Chất Đốt", res_cd, dt_cd_actual, isafe_cd, dynamic_risk_scores, current_display_dates)
render_tab(tab3, "Carbon", res_cb, dt_cb_actual, isafe_cb, dynamic_risk_scores, current_display_dates)