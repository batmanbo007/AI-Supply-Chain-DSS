import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fetch_real_weather import get_real_weather_7_days
from layer2_fuzzy_logic import build_fuzzy_system_thesis
from layer3_optimization import solve_milp_single_material


DEMAND_FILE = "D_t_forecast_7days.csv"
WEATHER_FILE = "Weather_Forecast_7days.csv"
APP_BUILD = "aligned-horizon-kpi-v3"


# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & NẠP ALIGNED HORIZON
# ==========================================
st.set_page_config(page_title="AI Supply Chain DSS", layout="wide")
st.title("🌪️ HỆ THỐNG QUẢN TRỊ CHUỖI CUNG ỨNG LAI (ML-FUZZY-MILP)")
st.markdown("*Tích hợp AI Dự báo Rủi ro Thời tiết & Tối ưu hóa Thời gian chờ động*")
st.caption(f"Build: {APP_BUILD}")


def load_aligned_inputs():
    """
    Dashboard không tự sinh ngày.
    Nó chỉ chạy khi Demand Forecast và Weather Forecast có đúng cùng 7 ngày.
    """
    try:
        demand = pd.read_csv(DEMAND_FILE)
        weather = pd.read_csv(WEATHER_FILE)
    except FileNotFoundError as exc:
        st.error(
            f"❌ Thiếu file đầu vào: {exc.filename}. "
            "Hãy chạy main_pipeline.py trước."
        )
        st.stop()
    except Exception as exc:
        st.error(f"❌ Không thể đọc dữ liệu pipeline: {exc}")
        st.stop()

    required_demand_cols = {
        "Ngay", "Dùng_Than", "Dùng_C.Đốt", "Dùng_Carbon"
    }
    required_weather_cols = {
        "Ngay", "Nhiet_Do_C", "Luong_Mua_mm", "Suc_Gio_kmh", "Bao"
    }

    missing_demand = required_demand_cols - set(demand.columns)
    missing_weather = required_weather_cols - set(weather.columns)

    if missing_demand:
        st.error(
            "❌ Demand Forecast thiếu cột: "
            + ", ".join(sorted(missing_demand))
        )
        st.stop()

    if missing_weather:
        st.error(
            "❌ Weather Forecast thiếu cột: "
            + ", ".join(sorted(missing_weather))
        )
        st.stop()

    demand["Ngay"] = pd.to_datetime(demand["Ngay"]).dt.normalize()
    weather["Ngay"] = pd.to_datetime(weather["Ngay"]).dt.normalize()

    if len(demand) != 7 or len(weather) != 7:
        st.error(
            "❌ Demand Forecast và Weather Forecast phải có đúng 7 ngày. "
            "Hãy chạy lại main_pipeline.py."
        )
        st.stop()

    if demand["Ngay"].duplicated().any() or weather["Ngay"].duplicated().any():
        st.error("❌ Phát hiện ngày trùng lặp trong dữ liệu pipeline.")
        st.stop()

    demand_dates = pd.DatetimeIndex(demand["Ngay"])
    weather_dates = pd.DatetimeIndex(weather["Ngay"])

    if not demand_dates.equals(weather_dates):
        st.error(
            "❌ Demand Forecast và Weather Forecast không cùng khoảng ngày. "
            "Dashboard dừng để tránh ra quyết định trên dữ liệu lệch thời gian."
        )
        st.stop()

    try:
        aligned = demand.merge(
            weather,
            on="Ngay",
            how="inner",
            validate="one_to_one"
        )
    except Exception as exc:
        st.error(f"❌ Không thể ghép Demand và Weather theo ngày: {exc}")
        st.stop()

    if len(aligned) != 7:
        st.error("❌ Sau khi merge theo Ngay không còn đủ 7 ngày.")
        st.stop()

    return demand, weather, aligned


def weather_to_ui_frame(weather_df):
    ui = weather_df[
        ["Ngay", "Nhiet_Do_C", "Luong_Mua_mm", "Suc_Gio_kmh", "Bao"]
    ].copy()

    ui["Ngay"] = pd.to_datetime(ui["Ngay"]).dt.strftime("%Y-%m-%d")

    return ui.rename(columns={
        "Ngay": "Ngày",
        "Nhiet_Do_C": "Nhiệt Độ (C)",
        "Luong_Mua_mm": "Lượng Mưa (mm)",
        "Suc_Gio_kmh": "Sức Gió (km/h)",
        "Bao": "Có Bão (1/0)",
    })


df_demand, df_weather, df_aligned = load_aligned_inputs()

forecast_dates = df_aligned["Ngay"].tolist()
display_dates = [
    date.strftime("%Y-%m-%d")
    for date in forecast_dates
]

st.info(
    "📅 **Decision Horizon:** "
    f"{forecast_dates[0].strftime('%d/%m/%Y')} → "
    f"{forecast_dates[-1].strftime('%d/%m/%Y')} · "
    "Demand và Weather đã được đồng bộ 1:1 theo ngày."
)
st.markdown("---")

dt_than = df_aligned["Dùng_Than"].tolist()
dt_cd = df_aligned["Dùng_C.Đốt"].tolist()
dt_cb = df_aligned["Dùng_Carbon"].tolist()

try:
    weather_ai = joblib.load("weather_risk_model.pkl")
except Exception as exc:
    st.error(f"❌ Không thể tải 'weather_risk_model.pkl': {exc}")
    st.stop()


# ==========================================
# 2. ĐIỀU ĐỘ NĂNG SUẤT DÂY CHUYỀN
# ==========================================
st.sidebar.header("🏭 ĐIỀU ĐỘ SẢN XUẤT")
st.sidebar.info(
    "Kéo thanh trượt để phân bổ linh hoạt công suất. "
    "(Mặc định hệ thống AI dự báo ở mốc Tổng 100%)"
)

cap_line1 = st.sidebar.slider(
    "🟢 Công suất Dây chuyền 1 (%)",
    min_value=0,
    max_value=150,
    value=60,
    step=10
)
cap_line2 = st.sidebar.slider(
    "🟢 Công suất Dây chuyền 2 (%)",
    min_value=0,
    max_value=150,
    value=40,
    step=10
)

total_cap = cap_line1 + cap_line2
st.sidebar.markdown(f"**🔥 Tổng công suất thực tế:** `{total_cap}%`")


def calculate_actual_demand(dt_list, c1, c2):
    total_ratio = (c1 + c2) / 100.0
    return [d * total_ratio for d in dt_list]


dt_than_actual = calculate_actual_demand(dt_than, cap_line1, cap_line2)
dt_cd_actual = calculate_actual_demand(dt_cd, cap_line1, cap_line2)
dt_cb_actual = calculate_actual_demand(dt_cb, cap_line1, cap_line2)


# ==========================================
# 3. THÔNG SỐ LOGISTICS & KHO BÃI
# ==========================================
st.sidebar.header("⚙️ THÔNG SỐ LOGISTICS & KHO BÃI")

with st.sidebar.expander("⚫ 1. KHO THAN", expanded=True):
    i0_than = st.number_input(
        "Tồn kho Than (I_0):",
        min_value=0,
        max_value=50000,
        value=3000,
        step=500
    )
    transport_than = st.selectbox(
        "Phương tiện vận tải (Than):",
        ["Xà lan (Đường thủy)", "Xe tải (Đường bộ)"]
    )
    lt_than = 3 if "Xà lan" in transport_than else 1
    st.caption(f"Thời gian chờ gốc (Lead time): **{lt_than} ngày**")

    co_than = (
        st.slider(
            "Cước xe Than (Triệu):",
            min_value=5,
            max_value=50,
            value=15
        )
        * 1000000
    )
    ch_than, imax_than = 25000, 50000

with st.sidebar.expander("🔥 2. KHO CHẤT ĐỐT", expanded=False):
    i0_cd = st.number_input(
        "Tồn kho C.Đốt (I_0):",
        min_value=0,
        max_value=20000,
        value=1000,
        step=200
    )
    lt_cd = st.slider(
        "Thời gian chờ (Lead Time) - Ngày:",
        min_value=0,
        max_value=5,
        value=1
    )
    co_cd = (
        st.slider(
            "Cước xe C.Đốt (Triệu):",
            min_value=2,
            max_value=30,
            value=8
        )
        * 1000000
    )
    ch_cd, imax_cd = 15000, 20000

with st.sidebar.expander("🪨 3. KHO CARBON", expanded=False):
    i0_cb = st.number_input(
        "Tồn kho Carbon (I_0):",
        min_value=0,
        max_value=5000,
        value=100,
        step=50
    )
    lt_cb = st.slider(
        "Thời gian chờ (Lead Time) C.B - Ngày:",
        min_value=0,
        max_value=5,
        value=0
    )
    co_cb = (
        st.slider(
            "Cước xe Carbon (Triệu):",
            min_value=1,
            max_value=20,
            value=5
        )
        * 1000000
    )
    ch_cb, imax_cb = 40000, 5000


# ==========================================
# 4. WEATHER + AI RISK, LUÔN GIỮ CÙNG HORIZON
# ==========================================
st.subheader("🌤️ 1. Dữ liệu Thời tiết & AI Dự báo Rủi ro")

expected_date_strings = display_dates

if "weather_data" not in st.session_state:
    st.session_state.weather_data = weather_to_ui_frame(df_weather)
else:
    session_dates = (
        st.session_state.weather_data.get("Ngày", pd.Series(dtype=str))
        .astype(str)
        .tolist()
    )
    if session_dates != expected_date_strings:
        # Demand horizon đã đổi sau một lần chạy pipeline mới:
        # reset weather session để không giữ dữ liệu từ horizon cũ.
        st.session_state.weather_data = weather_to_ui_frame(df_weather)

col_btn, col_info = st.columns([1, 4])

with col_btn:
    if st.button(
        "📡 Làm mới Thời Tiết (Hạ Long)",
        use_container_width=True
    ):
        with st.spinner("Đang tải weather cho đúng Decision Horizon..."):
            refreshed = get_real_weather_7_days(
                forecast_dates=forecast_dates,
                save_path=WEATHER_FILE
            )

            if refreshed is None:
                st.error(
                    "Không cập nhật weather vì API không trả đủ đúng "
                    "7 ngày của Decision Horizon."
                )
            else:
                refreshed_dates = refreshed["Ngay"].astype(str).tolist()

                if refreshed_dates != expected_date_strings:
                    st.error(
                        "Weather mới không khớp Demand Horizon. "
                        "Dữ liệu cũ được giữ nguyên."
                    )
                else:
                    st.session_state.weather_data = weather_to_ui_frame(
                        refreshed
                    )
                    st.success(
                        "Weather đã được làm mới và vẫn khớp 1:1 "
                        "với Demand Horizon."
                    )

with col_info:
    st.info(
        "💡 Ngày trong bảng bị khóa. Bạn có thể chỉnh các biến thời tiết "
        "để chạy kịch bản, nhưng Risk/Fuzzy/MILP luôn giữ nguyên "
        "7 ngày của Demand Forecast."
    )

edited_weather = st.data_editor(
    st.session_state.weather_data,
    num_rows="fixed",
    disabled=["Ngày"],
    use_container_width=True
)

if edited_weather["Ngày"].astype(str).tolist() != expected_date_strings:
    st.error("❌ Weather horizon đã bị thay đổi ngoài ý muốn.")
    st.stop()

X_predict = pd.DataFrame({
    "Nhiet_Do_C": pd.to_numeric(
        edited_weather["Nhiệt Độ (C)"],
        errors="coerce"
    ),
    "Luong_Mua_mm": pd.to_numeric(
        edited_weather["Lượng Mưa (mm)"],
        errors="coerce"
    ),
    "Suc_Gio_kmh": pd.to_numeric(
        edited_weather["Sức Gió (km/h)"],
        errors="coerce"
    ),
    "Bao": pd.to_numeric(
        edited_weather["Có Bão (1/0)"],
        errors="coerce"
    ),
})

if X_predict.isna().any().any():
    st.error("❌ Weather scenario có giá trị không hợp lệ.")
    st.stop()

dynamic_risk_scores = weather_ai.predict(X_predict)

cols = st.columns(7)
for i, col in enumerate(cols):
    risk_val = dynamic_risk_scores[i]
    color = (
        "#d63031"
        if risk_val >= 7
        else "#fdcb6e"
        if risk_val >= 4
        else "#00b894"
    )

    short_date = pd.Timestamp(forecast_dates[i]).strftime("%d/%m")
    col.markdown(
        "<div style='text-align: center; padding: 10px; "
        "background-color: #f1f2f6; border-radius: 5px; "
        f"border-bottom: 5px solid {color};'>"
        f"<b>{short_date}</b><br>"
        f"<h3 style='margin:0; color:{color}'>{risk_val:.1f}</h3>"
        "</div>",
        unsafe_allow_html=True
    )

st.markdown("---")


# ==========================================
# 5. FUZZY + MILP TRÊN CÙNG 7 NGÀY
# ==========================================
fuzzy_sim = build_fuzzy_system_thesis(visualize=False)


def calculate_dynamic_isafe(dt_list, max_cap):
    isafe_list = []

    for dt, risk in zip(dt_list, dynamic_risk_scores):
        if dt == 0:
            isafe_list.append(0)
        else:
            demand_pct = min(
                100,
                max(0, (dt / max_cap) * 100)
            )
            fuzzy_sim.input["Demand"] = demand_pct
            fuzzy_sim.input["Risk"] = min(10.0, float(risk))
            fuzzy_sim.compute()

            isafe_list.append(
                dt * fuzzy_sim.output["Safety_Days"]
            )

    return isafe_list


isafe_than = calculate_dynamic_isafe(dt_than_actual, 1000.0)
isafe_cd = calculate_dynamic_isafe(dt_cd_actual, 500.0)
isafe_cb = calculate_dynamic_isafe(dt_cb_actual, 200.0)

res_than = solve_milp_single_material(
    "Than",
    dt_than_actual,
    isafe_than,
    i0_than,
    co_than,
    ch_than,
    imax_than,
    base_lead_time=lt_than,
    risk_scores=dynamic_risk_scores
)
res_cd = solve_milp_single_material(
    "ChatDot",
    dt_cd_actual,
    isafe_cd,
    i0_cd,
    co_cd,
    ch_cd,
    imax_cd,
    base_lead_time=lt_cd,
    risk_scores=dynamic_risk_scores
)
res_cb = solve_milp_single_material(
    "Carbon",
    dt_cb_actual,
    isafe_cb,
    i0_cb,
    co_cb,
    ch_cb,
    imax_cb,
    base_lead_time=lt_cb,
    risk_scores=dynamic_risk_scores
)


# ==========================================
# 6. HIỂN THỊ KẾT QUẢ QUYẾT ĐỊNH
# ==========================================
total_system_cost = sum(
    res[1]
    for res in [res_than, res_cd, res_cb]
    if res[0]
)

st.subheader(
    "🏆 2. KẾT QUẢ TỐI ƯU HÓA "
    f"(ĐÃ TÍNH TRỄ HÀNG): {total_system_cost:,.0f} VNĐ"
)

if total_system_cost >= 1e9:
    st.error(
        "🚨 **BÁO ĐỘNG ĐỎ:** Hệ thống không thể nhập đủ hàng. "
        "Tồn đầu kỳ quá thấp so với thời gian chờ (Lead Time) "
        "hoặc có thời tiết rủi ro cao làm đứt gãy chuỗi cung ứng."
    )

tab1, tab2, tab3 = st.tabs([
    "⚫ Kịch bản: THAN",
    "🔥 Kịch bản: CHẤT ĐỐT",
    "🪨 Kịch bản: CARBON"
])


def render_tab(tab, name, res_data, dt, isafe, risk_arr, dates):
    with tab:
        success, cost, orders, q_vals, i_vals, r_vals, l_vals = res_data

        if not success:
            st.error(
                f"⚠️ Hệ thống MILP của {name} bị Infeasible."
            )
            return

        # Tạo đúng bảng hiển thị trước, sau đó tính KPI trực tiếp từ
        # chính các cột giao dịch của bảng này. Như vậy KPI và bảng không thể
        # dùng hai nguồn dữ liệu khác nhau.
        df_plot = pd.DataFrame({
            "Ngày": dates,
            "Nhu Cầu ($D_t$)": dt,
            "Thời Tiết (Rủi ro)": np.round(risk_arr, 1),
            "Chờ Hàng ($L_t$)": [
                f"{int(value)} ngày"
                for value in l_vals
            ],
            "Lệnh GỌI MUA ($Q_t$)": q_vals,
            "Hàng CẬP BẾN ($R_t$)": r_vals,
            "Tồn Thực Tế ($I_t$)": i_vals,
        })

        transaction_epsilon = 1e-6
        q_display = pd.to_numeric(
            df_plot["Lệnh GỌI MUA ($Q_t$)"],
            errors="coerce"
        ).fillna(0.0)
        r_display = pd.to_numeric(
            df_plot["Hàng CẬP BẾN ($R_t$)"],
            errors="coerce"
        ).fillna(0.0)

        order_count = int((q_display.abs() > transaction_epsilon).sum())
        arrival_count = int((r_display.abs() > transaction_epsilon).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Tổng Chi Phí",
            (
                f"{(cost % 1e9):,.0f} đ"
                if cost < 1e9
                else "RẤT LỚN (Phạt thiếu hàng)"
            )
        )
        c2.metric(
            "Số Lệnh Đặt Mua (Q_t > 0)",
            f"{order_count} chuyến"
        )
        c3.metric(
            "Số Lần Hàng Tới Bến (R_t > 0)",
            f"{arrival_count} chuyến"
        )

        # Kiểm tra nội bộ: nếu giá trị solver trả về khác KPI suy từ bảng,
        # dashboard vẫn ưu tiên bảng và phát cảnh báo để dễ debug.
        if int(orders) != order_count:
            st.warning(
                f"⚠️ KPI solver ({int(orders)}) khác số lệnh đang hiển thị "
                f"({order_count}). Dashboard dùng số từ Q_t hiển thị."
            )

        st.dataframe(
            df_plot.style.format(
                "{:.1f}",
                subset=[
                    "Nhu Cầu ($D_t$)",
                    "Thời Tiết (Rủi ro)",
                    "Lệnh GỌI MUA ($Q_t$)",
                    "Hàng CẬP BẾN ($R_t$)",
                    "Tồn Thực Tế ($I_t$)",
                ]
            ),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            dates,
            i_vals,
            marker="o",
            color="blue",
            linewidth=2,
            label="Tồn kho thực tế ($I_t$)"
        )
        ax.plot(
            dates,
            isafe,
            linestyle="--",
            color="red",
            linewidth=2,
            label="Đường an toàn ($I_{safe}$)"
        )

        ax2 = ax.twinx()
        width = 0.35
        x = np.arange(len(dates))

        ax2.bar(
            x - width / 2,
            q_vals,
            width,
            alpha=0.6,
            color="orange",
            label="📞 Lệnh Đặt Mua ($Q_t$)"
        )
        ax2.bar(
            x + width / 2,
            r_vals,
            width,
            alpha=0.6,
            color="green",
            label="🚢 Tàu Cập Bến ($R_t$)"
        )

        ax2.set_xticks(x)
        ax2.set_xticklabels(dates, rotation=30, ha="right")
        ax.set_ylabel("Khối lượng Tồn (Tấn)")
        ax2.set_ylabel("Khối lượng Giao Dịch (Tấn)")
        ax.set_title(
            "MÔ PHỎNG LỊCH TRÌNH ĐẶT HÀNG TRƯỚC "
            f"(DYNAMIC LEAD TIME) - {name.upper()}",
            fontweight="bold"
        )

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            lines + lines2,
            labels + labels2,
            loc="upper left"
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


render_tab(
    tab1,
    "Than",
    res_than,
    dt_than_actual,
    isafe_than,
    dynamic_risk_scores,
    display_dates
)
render_tab(
    tab2,
    "Chất Đốt",
    res_cd,
    dt_cd_actual,
    isafe_cd,
    dynamic_risk_scores,
    display_dates
)
render_tab(
    tab3,
    "Carbon",
    res_cb,
    dt_cb_actual,
    isafe_cb,
    dynamic_risk_scores,
    display_dates
)
