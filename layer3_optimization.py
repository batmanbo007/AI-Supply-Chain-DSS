import pandas as pd
import pulp
import warnings

warnings.filterwarnings('ignore')

def solve_milp_single_material(name, D_t, I_safe, I_0, C_o, C_h, I_max, base_lead_time=0, risk_scores=None):
    """
    Hàm giải MILP tích hợp Dynamic Lead Time (Thời gian giao hàng động)
    và Shortage Penalty (Chống đứt gãy chuỗi cung ứng).
    """
    days = list(range(1, len(D_t) + 1))
    model = pulp.LpProblem(f"Opt_{name}_Dynamic_LeadTime", pulp.LpMinimize)
    
    # 1. TÍNH TOÁN DYNAMIC LEAD TIME (Thời gian trễ cho từng ngày)
    L = {}
    for t in days:
        risk = risk_scores[t-1] if risk_scores is not None else 0
        # Logic bão tố: Rủi ro >= 7 (Trễ 2 ngày), Rủi ro >= 4 (Trễ 1 ngày)
        delay = 2 if risk >= 7 else (1 if risk >= 4 else 0)
        L[t] = base_lead_time + delay
        
    # 2. KHAI BÁO BIẾN QUYẾT ĐỊNH
    y = pulp.LpVariable.dicts(f"Call_{name}", days, cat='Binary')           # Quyết định gọi xe (1/0)
    Q = pulp.LpVariable.dicts(f"Order_{name}", days, lowBound=0, cat='Continuous') # Số lượng ĐẶT
    I = pulp.LpVariable.dicts(f"Inv_{name}", [0] + days, lowBound=0, cat='Continuous') # Tồn kho
    Shortage = pulp.LpVariable.dicts(f"Shortage_{name}", days, lowBound=0, cat='Continuous') # Hàng mua khẩn cấp
    
    # Ràng buộc tồn kho ban đầu
    model += I[0] == I_0
    
    # 3. HÀM MỤC TIÊU: Chi phí Đặt hàng + Chi phí Lưu kho + Phạt đứt gãy (1 Tỷ VNĐ/tấn)
    PENALTY_COST = 1e9 
    model += pulp.lpSum([C_o * y[t] + C_h * I[t] + PENALTY_COST * Shortage[t] for t in days])
    
    M = I_max * 2 # Big-M
    
    # 4. RÀNG BUỘC TOÁN HỌC CỐT LÕI
    for d in days:
        # TÌM HÀNG VỀ (Received): Tổng các lệnh ĐẶT từ quá khứ (Ngày t) sao cho Ngày t + Trễ = Ngày d hiện tại
        R_d = pulp.lpSum([Q[t] for t in days if t + L[t] == d])
        
        idx = d - 1
        # Cân bằng vật chất: Tồn hôm nay = Tồn hôm qua + HÀNG VỀ HÔM NAY - Tiêu thụ + Mua khẩn cấp
        model += I[d] == I[d-1] + R_d - D_t[idx] + Shortage[d]
        
        # Ràng buộc sức chứa & An toàn
        model += I[d] >= I_safe[idx]
        model += I[d] <= I_max
        
        # Ràng buộc Đặt hàng
        model += Q[d] <= M * y[d]
        
    # 5. GIẢI BÀI TOÁN
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[model.status] != 'Optimal':
        return False, 0, 0, [], [], [], []
        
    total_cost = pulp.value(model.objective)
    orders = sum(y[t].varValue for t in days)
    Q_vals = [Q[t].varValue for t in days]
    I_vals = [I[t].varValue for t in days]
    
    # Rút trích Hàng Về và Thời gian Trễ để báo cáo
    R_vals = []
    L_vals = []
    for d in days:
        received = sum(Q[t].varValue for t in days if t + L[t] == d)
        R_vals.append(received)
        L_vals.append(L[d])
        
    # CHÚ Ý: Hàm trả về 7 tham số thay vì 5 như trước (Thêm Hàng Về và Lead Time)
    return True, total_cost, orders, Q_vals, I_vals, R_vals, L_vals

# (Phần run_optimization_and_sensitivity cũ đã được ẩn đi vì chúng ta hiện tại chạy qua Dashboard)