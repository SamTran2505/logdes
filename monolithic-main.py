import gurobipy as gp
from gurobipy import GRB
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# ============================================================================
# PHẦN 1: KHỞI TẠO THAM SỐ VÀ DỮ LIỆU (ĐHQG TPHCM)
# ============================================================================
M = 2  # Nhà sản xuất
J = 4  # Candidate CDCs
R = 4  # Candidate RDCs
K = 2  # Khách hàng
T = 12 # Chu kỳ
N = J + R
N_scenarios = N + 1

# Tọa độ thực tế (Vĩ độ, Kinh độ) tại khu đô thị ĐHQG TPHCM
coords_M = np.array([
    [10.8806, 106.8054], # 0: ĐH Bách Khoa (HCMUT)
    [10.8700, 106.8030]  # 1: ĐH Công nghệ Thông tin (UIT)
])

coords_J = np.array([
    [10.8801, 106.7978], # 0: ĐH Quốc Tế (HCMIU) - Sân nhà
    [10.8795, 106.7898], # 1: Nhà điều hành ĐHQG
    [10.8705, 106.7750], # 2: ĐH Kinh tế - Luật (UEL)
    [10.8746, 106.7997]  # 3: Thư viện Trung tâm
])

coords_R = np.array([
    [10.8761, 106.7972], # 0: ĐH Khoa học Tự nhiên (HCMUS)
    [10.8716, 106.8028], # 1: ĐH Khoa học Xã hội & Nhân văn (USSH)
    [10.8835, 106.7925], # 2: Khoa Y ĐHQG
    [10.8932, 106.7961]  # 3: TT Giáo dục Quốc phòng & An ninh
])

coords_K = np.array([
    [10.8783, 106.8063], # 0: Ký túc xá Khu A
    [10.8850, 106.7825]  # 1: Ký túc xá Khu B
])

coords_N = np.vstack((coords_J, coords_R))

def calc_dist(c1, c2):
    R_earth = 6371.0 
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R_earth * c

# Đã điều chỉnh lại thành 5 để chi phí giao hàng chiếm khoảng 9% giá sản phẩm
unit_trans_cost = 5

# Giá vốn hóa dựa trên thực tế Dĩ An, Bình Dương
f_j = np.random.uniform(90000, 100000, J)
f_r = np.random.uniform(75000, 85000, R)

L_n = np.concatenate((np.full(J, 4500.0), np.full(R, 2000.0)))
d_k = np.random.uniform(1800, 2200, (K, T))  
v_k = np.random.uniform(0.2, 0.3, (K, T)) * d_k
h_n = np.concatenate((np.random.uniform(1.2, 1.6, J), np.random.uniform(0.8, 1.2, R)))

epicenter = np.array([10.8767, 106.8000])
theta = 3
D_n = np.array([calc_dist(coords_N[n], epicenter) for n in range(N)])

# Bộ thông số tối ưu kinh tế học
rho = 100
rho_rev = 50
psi = 50
zeta = 20

prob_s = None
safety_rank = None

def update_risk_probabilities(current_tau):
    global prob_s, safety_rank
    q_raw = current_tau * np.exp(-D_n / theta)
    safety_rank = np.argsort(q_raw)
    q_sorted = np.sort(q_raw)
    q = np.insert(q_sorted, 0, 0.0) 
    q_extended = np.append(q, 1.0)
    prob_s = np.diff(q_extended) 

# ============================================================================
# PHẦN 2: BÀI TOÁN CON (Subproblem)
# ============================================================================
def solve_subproblem(s, x_star, y_star, mode="hybrid"):
    sp = gp.Model(f"Sub_{s}_{mode}")
    sp.Params.OutputFlag = 0
    sp.Params.QCPDual = 1 
    
    lam = sp.addVars(N, T, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    gam = sp.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0, ub=1)
    Q_inv = sp.addVars(N, T, vtype=GRB.CONTINUOUS, lb=0)
    Y_inv = sp.addVars(J, T, vtype=GRB.CONTINUOUS, lb=0) 
    w_0k = sp.addVars(K, T, vtype=GRB.CONTINUOUS, lb=0) # Biến thiếu hụt luồng đi (w_0k)
    u_0k = sp.addVars(K, T, vtype=GRB.CONTINUOUS, lb=0) # Biến thiếu hụt luồng về (u_0k)
    b = sp.addVars(M, T, vtype=GRB.CONTINUOUS, lb=0)
    u = sp.addVars(M, T, vtype=GRB.CONTINUOUS, lb=0)
    p_forward = sp.addVars(M, J, T, vtype=GRB.CONTINUOUS, lb=0)
    p_down = sp.addVars(J, K, T, vtype=GRB.CONTINUOUS, lb=0)
    l_reverse = sp.addVars(K, N, T, vtype=GRB.CONTINUOUS, lb=0)
    l_up = sp.addVars(N, M, T, vtype=GRB.CONTINUOUS, lb=0)

    if mode == "rdc_only":
        sp.addConstrs((lam[j, t] == 0 for j in range(J) for t in range(T)), name="Force_RDC_only")

    constr_alpha = sp.addConstrs((lam[j, t] + gam[j, t] <= x_star[j] for j in range(J) for t in range(T)), name="alpha")
    constr_beta = sp.addConstrs((lam[J+r, t] <= y_star[r] for r in range(R) for t in range(T)), name="beta")

    survivors = safety_rank[:s] if s > 0 else []
    if s == N_scenarios - 1: survivors = range(N)
    xi_s = [1 if n in survivors else 0 for n in range(N)]
    
    sp.addConstrs((Q_inv[n, t] <= L_n[n] * lam[n, t] * xi_s[n] for n in range(N) for t in range(T)))
    sp.addConstrs((Y_inv[j, t] <= L_n[j] * gam[j, t] * xi_s[j] for j in range(J) for t in range(T)))
    
    for j in range(J):
        for t in range(T):
            sp.addConstr(gp.quicksum(p_forward[m, j, t] for m in range(M)) <= L_n[j] * gam[j, t] * xi_s[j])       
    for n in range(N):
        for t in range(T):
            sp.addConstr(gp.quicksum(l_reverse[k, n, t] for k in range(K)) <= L_n[n] * lam[n, t] * xi_s[n])
            
    for n in range(N):
        for t in range(T):
            flow_in = gp.quicksum(l_reverse[k, n, t] for k in range(K))
            flow_out = gp.quicksum(l_up[n, m, t] for m in range(M))
            if t == 0: sp.addConstr(Q_inv[n, t] == flow_in - flow_out)
            else: sp.addConstr(Q_inv[n, t] == Q_inv[n, t-1] + flow_in - flow_out)
                
    for j in range(J):
        for t in range(T):
            flow_in = gp.quicksum(p_forward[m, j, t] for m in range(M))
            flow_out = gp.quicksum(p_down[j, k, t] for k in range(K))
            if t == 0: sp.addConstr(Y_inv[j, t] == flow_in - flow_out)
            else: sp.addConstr(Y_inv[j, t] == Y_inv[j, t-1] + flow_in - flow_out)

    for t in range(T):
        for k in range(K):
            sp.addConstr(gp.quicksum(p_down[j, k, t] for j in range(J)) + w_0k[k, t] >= d_k[k, t])
            sp.addConstr(gp.quicksum(l_reverse[k, n, t] for n in range(N)) + u_0k[k, t] == v_k[k, t])
        for m in range(M):
            sp.addConstr(b[m, t] + u[m, t] >= gp.quicksum(p_forward[m, j, t] for j in range(J)))
            sp.addConstr(u[m, t] <= gp.quicksum(l_up[n, m, t] for n in range(N)))

    TC_reverse_CDC = gp.quicksum(calc_dist(coords_K[k], coords_N[j]) * l_reverse[k, j, t] for k in range(K) for j in range(J) for t in range(T))
    TC_reverse_RDC = gp.quicksum(calc_dist(coords_K[k], coords_N[J+r]) * l_reverse[k, J+r, t] for k in range(K) for r in range(R) for t in range(T))

    TC = unit_trans_cost * (gp.quicksum(calc_dist(coords_K[k], coords_N[j]) * l_reverse[k, j, t] for k in range(K) for j in range(J) for t in range(T)) + \
         gp.quicksum(calc_dist(coords_N[n], coords_M[m]) * l_up[n,m,t] for n in range(N) for m in range(M) for t in range(T)) + \
         gp.quicksum(calc_dist(coords_M[m], coords_J[j]) * p_forward[m,j,t] for m in range(M) for j in range(J) for t in range(T)) + \
         gp.quicksum(calc_dist(coords_J[j], coords_K[k]) * p_down[j,k,t] for j in range(J) for k in range(K) for t in range(T)))
         
    IC = gp.quicksum(h_n[n] * Q_inv[n,t] for n in range(N) for t in range(T)) + gp.quicksum(h_n[j] * Y_inv[j,t] for j in range(J) for t in range(T))
    LC = rho * gp.quicksum(w_0k[k,t] for k in range(K) for t in range(T)) + rho_rev * gp.quicksum(u_0k[k,t] for k in range(K) for t in range(T))
    PC = gp.quicksum(psi * b[m,t] + zeta * u[m,t] for m in range(M) for t in range(T))
    
    sp.setObjective(TC + IC + LC + PC, GRB.MINIMIZE)
    sp.optimize()
    
    if sp.Status == GRB.OPTIMAL:
        alpha_duals = {(j, t): constr_alpha[j, t].Pi for j in range(J) for t in range(T)}
        beta_duals = {(r, t): constr_beta[r, t].Pi for r in range(R) for t in range(T)}
        
        # ---> SỬA Ở ĐÂY: Trích xuất trực tiếp khối lượng rớt đơn để tính DS và RR
        sum_w = sum(w_0k[k, t].X for k in range(K) for t in range(T))
        sum_u = sum(u_0k[k, t].X for k in range(K) for t in range(T))
        
        return True, sp.ObjVal, alpha_duals, beta_duals, sum_w, sum_u
    return False, None, None, None, 0, 0

def calculate_kappa(s, x_sol, y_sol, node_idx, is_cdc=True, mode="hybrid"):
    x_temp, y_temp = list(x_sol), list(y_sol)
    if is_cdc: x_temp[node_idx] = 0.0
    else: y_temp[node_idx] = 0.0
    
    eps = 1e-4
    x_temp_relaxed = [max(x, eps) for x in x_temp]
    y_temp_relaxed = [max(y, eps) for y in y_temp]
    
    # Hứng thêm 2 biến thiếu hụt (nhưng không dùng ở đây)
    is_feas, q_s_new, _, _, _, _ = solve_subproblem(s, x_temp_relaxed, y_temp_relaxed, mode)
    return q_s_new if is_feas else 9999999.0

# ============================================================================
# PHẦN 3: MASTER PROBLEM VÀ CALLBACK
# ============================================================================
def build_and_solve_master(mode="hybrid"):
    mp = gp.Model(f"Master_{mode}")
    mp.Params.OutputFlag = 0
    mp.Params.LazyConstraints = 1
    
    x = mp.addVars(J, vtype=GRB.BINARY, name="x_CDC")
    y = mp.addVars(R, vtype=GRB.BINARY, name="y_RDC")
    Phi = mp.addVars(N_scenarios, lb=0, vtype=GRB.CONTINUOUS, name="Phi")
    
    if mode == "cdc_only":
        mp.addConstrs((y[r] == 0 for r in range(R)), name="Force_CDC_only")
        
    mp.setObjective(gp.quicksum(f_j[j] * x[j] for j in range(J)) + gp.quicksum(f_r[r] * y[r] for r in range(R)) + gp.quicksum(Phi[s] for s in range(N_scenarios)), GRB.MINIMIZE)
    mp._x, mp._y, mp._Phi = x, y, Phi
    
    def bbcd_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            try:
                x_sol = model.cbGetSolution(model._x)
                y_sol = model.cbGetSolution(model._y)
                Phi_sol = model.cbGetSolution(model._Phi)
                
                eps = 1e-4
                x_sol_relaxed = [max(x_sol[j], eps) for j in range(J)]
                y_sol_relaxed = [max(y_sol[r], eps) for r in range(R)]
                
                for s in range(N_scenarios):
                    # Hứng thêm 2 biến thiếu hụt (nhưng không dùng ở đây)
                    is_feasible, q_s, alpha, beta, _, _ = solve_subproblem(s, x_sol_relaxed, y_sol_relaxed, mode)
                    
                    if not is_feasible:
                        cut_expr = gp.quicksum(1 - model._x[j] for j in range(J) if x_sol[j] > 0.5) + \
                                   gp.quicksum(model._x[j] for j in range(J) if x_sol[j] <= 0.5) + \
                                   gp.quicksum(1 - model._y[r] for r in range(R) if y_sol[r] > 0.5) + \
                                   gp.quicksum(model._y[r] for r in range(R) if y_sol[r] <= 0.5)
                        model.cbLazy(cut_expr >= 1)
                        break 
                    else:
                        weighted_q_s = prob_s[s] * q_s
                        if Phi_sol[s] < weighted_q_s - 1e-4:
                            cut_expr_s = q_s
                            for j in range(J):
                                for t in range(T):
                                    cut_expr_s += alpha[j, t] * (model._x[j] - x_sol_relaxed[j])
                            for r in range(R):
                                for t in range(T):
                                    cut_expr_s += beta[r, t] * (model._y[r] - y_sol_relaxed[r])
                            model.cbLazy(model._Phi[s] >= prob_s[s] * cut_expr_s)
            except Exception as e:
                print(f"!!! LỖI CALLBACK: {e}")
    mp.optimize(bbcd_callback)
    if mp.SolCount > 0:
        x_opt = [x[j].X for j in range(J)]
        y_opt = [y[r].X for r in range(R)]
        return mp.ObjVal, x_opt, y_opt  
    return float('inf'), [], []

# ============================================================================
# PHẦN 4: THỰC THI (MAIN) - LẶP TỪNG MỨC TAU VÀ VẼ BIỂU ĐỒ
# ============================================================================
if __name__ == "__main__":
    print("="*95)
    print("      SO SÁNH HYBRID vs CDC-ONLY vs RDC-ONLY KÈM PHÂN TÍCH SERVICE LEVEL")
    print("="*95)
    
    tau_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    hybrid_costs, cdc_costs, rdc_costs = [], [], []
    hybrid_ds_list, hybrid_rr_list = [], [] # Lưu trữ DS và RR để vẽ biểu đồ
    
    # Tính tổng nhu cầu của toàn hệ thống trong suốt T chu kỳ để làm mẫu số
    total_demand = np.sum(d_k)
    total_returns = np.sum(v_k)
    
    # Header Terminal (Đã được mở rộng)
    print(f"{'Tau':<4} | {'Hybrid Cost':<12} | {'CDC-Only':<12} | {'RDC-Only':<12} | {'Structure of Hybrid':<16} | {'DS (%)':<7} | {'RR (%)':<7}")
    print("-" * 95)
    
    for current_tau in tau_levels:
        update_risk_probabilities(current_tau)
        
        cost_hybrid, x_opt, y_opt = build_and_solve_master(mode="hybrid")
        cost_cdc, _, _ = build_and_solve_master(mode="cdc_only")
        cost_rdc, _, _ = build_and_solve_master(mode="rdc_only")
        
        num_cdc = int(sum(x_opt)) if cost_hybrid != float('inf') else 0
        num_rdc = int(sum(y_opt)) if cost_hybrid != float('inf') else 0
        
        hybrid_costs.append(cost_hybrid)
        cdc_costs.append(cost_cdc)
        rdc_costs.append(cost_rdc)
        
        # ---> TÍNH TOÁN EXPECTED DS VÀ RR SAU KHI CÓ NGHIỆM TỐI ƯU
        expected_shortage_forward = 0
        expected_shortage_reverse = 0
        
        if cost_hybrid != float('inf'):
            for s in range(N_scenarios):
                is_feas, _, _, _, s_w, s_u = solve_subproblem(s, [max(x, 1e-4) for x in x_opt], [max(y, 1e-4) for y in y_opt], mode="hybrid")
                if is_feas:
                    expected_shortage_forward += prob_s[s] * s_w
                    expected_shortage_reverse += prob_s[s] * s_u
                    
        DS = 100 * (1 - expected_shortage_forward / total_demand) if total_demand > 0 else 100
        RR = 100 * (1 - expected_shortage_reverse / total_returns) if total_returns > 0 else 100
        
        hybrid_ds_list.append(DS)
        hybrid_rr_list.append(RR)
        
        # In ra màn hình mượt mà
        print(f"{current_tau:<4.1f} | {cost_hybrid:<12,.0f} | {cost_cdc:<12,.0f} | {cost_rdc:<12,.0f} | Open: {num_cdc} CDC, {num_rdc} RDC | {DS:>6.2f}% | {RR:>6.2f}%")
    
    print("="*95)
    print("Thuật toán đã giải xong! Đang tiến hành vẽ 2 biểu đồ...")

    # --- BIỂU ĐỒ 1: CHI PHÍ ---
    hybrid_m = [c / 1e6 for c in hybrid_costs]
    cdc_m = [c / 1e6 for c in cdc_costs]
    rdc_m = [c / 1e6 for c in rdc_costs]

    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'serif' 

    plt.plot(tau_levels, cdc_m, marker='o', linestyle=':', color='royalblue', linewidth=2, label='CDC-Only')
    plt.plot(tau_levels, rdc_m, marker='s', linestyle='-.', color='indianred', linewidth=2, label='RDC-Only')
    plt.plot(tau_levels, hybrid_m, marker='*', linestyle='--', color='darkgreen', linewidth=2, markersize=10, label='Hybrid')

    plt.title('Total Cost Comparison Under Varying Disruption Risk Levels', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Core Disruption Risk Probability (Tau)', fontsize=12)
    plt.ylabel('Total Cost (Millions USD)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper left')
    plt.xticks(tau_levels)
    plt.tight_layout()
    plt.savefig('Experiment_1_Cost_Comparison.png', dpi=300, bbox_inches='tight')

    # --- BIỂU ĐỒ 2: DEMAND SATISFACTION VÀ RECYCLING RATE ---
    plt.figure(figsize=(10, 5))
    plt.plot(tau_levels, hybrid_ds_list, marker='o', color='forestgreen', linewidth=2, label='Demand Satisfaction (DS)')
    plt.plot(tau_levels, hybrid_rr_list, marker='s', color='darkorange', linewidth=2, label='Recycling Rate (RR)')
    
    plt.title('Service Level & CSR Performance of Hybrid Network', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Disruption Risk Probability (Tau)', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.ylim(0, 105) # Giữ cho trục tung y luôn hiển thị từ 0 tới 100%
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower left')
    plt.xticks(tau_levels)
    plt.tight_layout()
    plt.savefig('Experiment_2_Service_Levels.png', dpi=300, bbox_inches='tight')
    
    print(">> [THÀNH CÔNG] Đã lưu cả 2 biểu đồ vào máy.")
    plt.show()