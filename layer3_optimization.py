import pandas as pd
import pulp
import warnings

warnings.filterwarnings('ignore')

def solve_milp_single_material(name, D_t, I_safe, I_0, C_o, C_h, I_max, base_lead_time=0, risk_scores=None):
    """
    Hàm giải MILP tích hợp Dynamic Lead Time và Padding Time Horizon
    để khắc phục hiệu ứng "Chân trời hữu hạn" và "Chi phí lưu kho ảo".
    """
    # ========================================================
    # BƯỚC A: KỸ THUẬT PADDING (MỞ RỘNG CHÂN TRỜI TÍNH TOÁN)
    # ========================================================
    N = len(D_t) # Số ngày thực tế (Thường là 7)
    pad_len = 7  # Mở rộng thêm 7 ngày ảo để AI không ngừng mua ở cuối chu kỳ
    total_days = N + pad_len
    
    # Nội suy dữ liệu cho 7 ngày ảo (Giữ nguyên mức tiêu dùng ngày cuối cùng, thời tiết bình thường)
    D_t_padded = D_t + [D_t[-1]] * pad_len
    I_safe_padded = I_safe + [I_safe[-1]] * pad_len
    risk_padded = list(risk_scores) + [0] * pad_len if risk_scores is not None else [0] * total_days
    
    days = list(range(1, total_days + 1))
    model = pulp.LpProblem(f"Opt_{name}_Dynamic_LeadTime_Padded", pulp.LpMinimize)
    
    # 1. TÍNH TOÁN DYNAMIC LEAD TIME (Trên trục thời gian 14 ngày)
    L = {}
    for t in days:
        risk = risk_padded[t-1]
        delay = 2 if risk >= 7 else (1 if risk >= 4 else 0)
        L[t] = base_lead_time + delay
        
    # 2. KHAI BÁO BIẾN QUYẾT ĐỊNH
    y = pulp.LpVariable.dicts(f"Call_{name}", days, cat='Binary')           
    Q = pulp.LpVariable.dicts(f"Order_{name}", days, lowBound=0, cat='Continuous') 
    I = pulp.LpVariable.dicts(f"Inv_{name}", [0] + days, lowBound=0, cat='Continuous') 
    Shortage = pulp.LpVariable.dicts(f"Shortage_{name}", days, lowBound=0, cat='Continuous') 
    
    model += I[0] == I_0
    
    # ========================================================
    # BƯỚC B: HÀM MỤC TIÊU CẢI TIẾN (ĐỒNG BỘ CHI PHÍ)
    # ========================================================
    # Xử lý hiệu ứng "Lưu kho quá đắt": Tự động quy đổi C_h (phí năm) về Phí lưu kho NGÀY.
    # Nhờ vậy AI sẽ tự tin gộp hàng thành các chuyến lớn thay vì gọi xe lắt nhắt mỗi ngày.
    daily_C_h = C_h / 365.0 
    PENALTY_COST = 1e9 
    
    model += pulp.lpSum([C_o * y[t] + daily_C_h * I[t] + PENALTY_COST * Shortage[t] for t in days])
    
    M = I_max * 2 
    
    # 3. RÀNG BUỘC TOÁN HỌC CỐT LÕI
    for d in days:
        R_d = pulp.lpSum([Q[t] for t in days if t + L[t] == d])
        
        idx = d - 1
        model += I[d] == I[d-1] + R_d - D_t_padded[idx] + Shortage[d]
        model += I[d] >= I_safe_padded[idx]
        model += I[d] <= I_max
        model += Q[d] <= M * y[d]
        
    # 4. GIẢI BÀI TOÁN
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[model.status] != 'Optimal':
        return False, 0, 0, [], [], [], []
        
    # ========================================================
    # BƯỚC C: TRÍCH XUẤT ĐÚNG 7 NGÀY THỰC TẾ CHO DASHBOARD
    # ========================================================
    display_days = list(range(1, N + 1))
    
    # Tính lại Chi phí thực tế (Chỉ đếm tiền của 7 ngày thực để hiển thị KPI chuẩn)
    total_cost_N_days = sum(
        C_o * (y[t].varValue or 0.0) + 
        daily_C_h * (I[t].varValue or 0.0) + 
        PENALTY_COST * (Shortage[t].varValue or 0.0) 
        for t in display_days
    )

    ORDER_EPSILON = 1e-6
    Q_vals = [float(Q[t].varValue or 0.0) for t in display_days]
    orders = sum(1 for q in Q_vals if q > ORDER_EPSILON)

    I_vals = [float(I[t].varValue or 0.0) for t in display_days]
    
    R_vals = []
    L_vals = []
    for d in display_days:
        received = sum(
            float(Q[t].varValue or 0.0)
            for t in days # Vẫn phải quét mảng 14 ngày để không bỏ sót kiện hàng nào
            if t + L[t] == d
        )
        R_vals.append(received)
        L_vals.append(L[d])
        
    return True, total_cost_N_days, orders, Q_vals, I_vals, R_vals, L_vals