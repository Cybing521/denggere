#!/usr/bin/env python3
"""
完整蚊虫-人群动力学模型
参考: Zhang et al. (2025) PLOS NTD

特点:
1. 完整蚊虫生命周期 (卵→幼虫→蛹→成蚊)
2. 温度依赖参数 (来自文献公式)
3. 输入病例
4. 周尺度
"""
import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

log("=" * 65)
log("蚊虫-人群耦合动力学模型")
log("参考: Zhang et al. (2025) PLOS NTD")
log("=" * 65)

# ============================================================
# 温度依赖参数 (Table 2)
# ============================================================
class TempParams:
    @staticmethod
    def phi(T):  # 产卵率 (2)
        if T < 14 or T > 35: return 0
        return max(0, -0.02*T**2 + 1.1*T - 8.5)
    
    @staticmethod
    def f_e(T):  # 卵孵化率 (3)
        if T < 10 or T > 40: return 0
        return max(0, 0.0022*T*(T-10)*np.sqrt(40-T))
    
    @staticmethod
    def f_l(T):  # 幼虫发育率 (4)
        if T < 10 or T > 40: return 0
        return max(0, 0.00085*T*(T-10)*np.sqrt(40-T))
    
    @staticmethod
    def f_p(T):  # 蛹发育率 (5)
        if T < 10 or T > 40: return 0
        return max(0, 0.0021*T*(T-10)*np.sqrt(40-T))
    
    @staticmethod
    def mu_l(T):  # 幼虫死亡率 (6)
        if T < 10 or T > 40: return 1
        return max(0.01, 0.0025*T**2 - 0.094*T + 0.96)
    
    @staticmethod
    def mu_p(T):  # 蛹死亡率 (7)
        if T < 10 or T > 40: return 1
        return max(0.01, 0.0003*T**2 - 0.0126*T + 0.14)
    
    @staticmethod
    def mu_m(T):  # 成蚊死亡率 (8)
        if T < 10 or T > 40: return 1
        return max(0.02, 0.0006*T**2 - 0.028*T + 0.37)
    
    @staticmethod
    def a(T):  # 叮咬率 (9)
        if T < 14 or T > 35: return 0
        return max(0, 0.0005*T*(T-14)*np.sqrt(35-T))
    
    @staticmethod
    def b(T):  # 感染概率 M→H (10)
        if T < 17 or T > 36: return 0
        return max(0, 0.0008*T*(T-17)*np.sqrt(36-T))
    
    @staticmethod
    def c(T):  # 感染概率 H→M (11)
        if T < 12 or T > 37: return 0
        return max(0, 0.0007*T*(T-12)*np.sqrt(37-T))

# 固定参数
MU_E = 0.1      # 蚊卵死亡率
MU_EM = 0.1     # 羽化死亡率
SIGMA = 0.5     # 雌蚊比例
DELTA = 1/5     # 人潜伏期转化率 (5天)
GAMMA = 1/6     # 康复率 (6天)
N_H = 14_000_000

# ============================================================
# 加载数据
# ============================================================
log("\n[1] 加载数据...")

case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()

log(f"  月数据: {len(df)}个月")
log(f"  病例: {df['cases'].sum():,}")

# 周数据
n_weeks = len(df) * 4
weekly_temp = np.repeat(df['temperature'].values, 4)
weekly_cases = np.repeat(df['cases'].values / 4, 4)

log(f"  周数据: {n_weeks}周")
log(f"  温度范围: {weekly_temp.min():.1f} - {weekly_temp.max():.1f}°C")

# ============================================================
# 计算温度依赖参数时间序列
# ============================================================
log("\n[2] 计算温度依赖参数...")

p = TempParams()
phi_arr = np.array([p.phi(T) for T in weekly_temp])
f_e_arr = np.array([p.f_e(T) for T in weekly_temp])
f_l_arr = np.array([p.f_l(T) for T in weekly_temp])
f_p_arr = np.array([p.f_p(T) for T in weekly_temp])
mu_l_arr = np.array([p.mu_l(T) for T in weekly_temp])
mu_p_arr = np.array([p.mu_p(T) for T in weekly_temp])
mu_m_arr = np.array([p.mu_m(T) for T in weekly_temp])
a_arr = np.array([p.a(T) for T in weekly_temp])
b_arr = np.array([p.b(T) for T in weekly_temp])
c_arr = np.array([p.c(T) for T in weekly_temp])

log(f"  φ (产卵率): {phi_arr.min():.3f} - {phi_arr.max():.3f}")
log(f"  μ_m (成蚊死亡率): {mu_m_arr.min():.3f} - {mu_m_arr.max():.3f}")
log(f"  a (叮咬率): {a_arr.min():.3f} - {a_arr.max():.3f}")

# ============================================================
# 蚊虫动态模型 (不含人群)
# ============================================================
log("\n[3] 模拟蚊虫动态...")

def mosquito_ode(y, t, idx_arr):
    """蚊虫生命周期: E → L → P → M"""
    E, L, P, M = y
    
    # 当前时间索引
    idx = min(int(t), len(idx_arr) - 1)
    
    phi = phi_arr[idx]
    f_e = f_e_arr[idx]
    f_l = f_l_arr[idx]
    f_p = f_p_arr[idx]
    mu_l = mu_l_arr[idx]
    mu_p = mu_p_arr[idx]
    mu_m = mu_m_arr[idx]
    
    # 环境容纳量
    K = N_H * 3
    
    # 蚊虫动态
    dE = phi * M * SIGMA * (1 - E/K) - (f_e + MU_E) * E
    dL = f_e * E - (f_l + mu_l) * L
    dP = f_l * L - (f_p + mu_p) * P
    dM = f_p * P * (1 - MU_EM) - mu_m * M
    
    return [dE, dL, dP, dM]

# 模拟蚊虫
time_weeks = np.arange(n_weeks)
idx_arr = time_weeks.astype(int)

# 初始条件 (根据初始温度估计)
T0 = weekly_temp[0]
M0 = N_H * 0.01  # 初始成蚊数
E0 = M0 * 5
L0 = M0 * 3
P0 = M0 * 2

y0_mosq = [E0, L0, P0, M0]
sol_mosq = odeint(mosquito_ode, y0_mosq, time_weeks, args=(idx_arr,))

E_mosq, L_mosq, P_mosq, M_mosq = sol_mosq.T

log(f"  成蚊数量范围: {M_mosq.min():.2e} - {M_mosq.max():.2e}")

# ============================================================
# 交叉相关分析 (蚊虫 vs 病例)
# ============================================================
log("\n[4] 交叉相关分析...")

# 标准化
M_norm = (M_mosq - M_mosq.mean()) / (M_mosq.std() + 1e-10)
cases_norm = (weekly_cases - weekly_cases.mean()) / (weekly_cases.std() + 1e-10)

# 计算不同滞后的相关性
max_lag = 12  # 周
best_lag = 0
best_corr = -1

for lag in range(0, max_lag + 1):
    if lag == 0:
        corr, _ = pearsonr(M_norm, cases_norm)
    else:
        corr, _ = pearsonr(M_norm[:-lag], cases_norm[lag:])
    
    if corr > best_corr:
        best_corr = corr
        best_lag = lag

log(f"  最佳滞后: {best_lag} 周")
log(f"  最佳相关: {best_corr:.4f}")

# ============================================================
# 人群SEIR模型 (使用蚊虫数据)
# ============================================================
log("\n[5] 人群SEIR模型...")

def seir_ode(y, t, M_arr, idx_arr, b_h, imp):
    """人群SEIR，蚊虫数据作为输入"""
    S, E, I, R = y
    
    idx = min(int(t), len(idx_arr) - 1)
    
    M = M_arr[idx]
    a_t = a_arr[idx]
    b_t = b_arr[idx]
    c_t = c_arr[idx]
    
    # 感染蚊比例 (简化估计)
    i_v = c_t * a_t * I / N_H * 10  # 放大因子
    
    # 感染力
    lam = b_h * a_t * b_t * M * i_v / N_H
    
    # SEIR
    dS = -lam * S
    dE = lam * S + imp - DELTA * E
    dI = DELTA * E - GAMMA * I
    dR = GAMMA * I
    
    return [dS, dE, dI, dR]

# 参数搜索
log("  参数搜索...")

best_loss = 1e10
best_params = None
best_pred = None

for b_h in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
    for imp in [0, 0.5, 1, 2, 5, 10]:
        for I0_exp in [-6, -5, -4]:
            I0 = 10 ** I0_exp
            y0 = [N_H - I0*2, I0, I0, 0]
            
            try:
                sol = odeint(seir_ode, y0, time_weeks, args=(M_mosq, idx_arr, b_h, imp))
                I_h = sol[:, 2]
                pred = GAMMA * I_h * 7
                
                obs_log = np.log1p(weekly_cases)
                pred_log = np.log1p(pred)
                mse = np.mean((obs_log - pred_log) ** 2)
                
                if mse < best_loss:
                    best_loss = mse
                    best_params = (b_h, imp, I0)
                    best_pred = pred.copy()
            except:
                pass

b_h, imp, I0 = best_params
log(f"  b_h = {b_h}, imp = {imp}, I0 = {I0:.2e}")
log(f"  Loss = {best_loss:.4f}")

# 评估
corr, pval = pearsonr(weekly_cases, best_pred)
r2_log = r2_score(np.log1p(weekly_cases), np.log1p(best_pred))

log(f"\n  相关系数: {corr:.4f}")
log(f"  R² (对数): {r2_log:.4f}")

# ============================================================
# R0 计算
# ============================================================
# R0 = (b_h × a² × b × c × M) / (μ_m × γ × N_h) × 存活概率
R0_arr = []
for i in range(n_weeks):
    M = M_mosq[i]
    a_t = a_arr[i]
    b_t = b_arr[i]
    c_t = c_arr[i]
    mu_m = mu_m_arr[i]
    
    R0 = b_h * a_t**2 * b_t * c_t * M / (mu_m * GAMMA * N_H + 1e-10)
    R0_arr.append(R0)

R0_arr = np.array(R0_arr)
log(f"  R0 范围: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]")

# ============================================================
# 可视化
# ============================================================
log("\n[6] 可视化...")

fig = plt.figure(figsize=(18, 14))

# 1. 病例对比
ax = fig.add_subplot(3, 3, 1)
ax.plot(weekly_cases, 'b-', lw=1, label='Observed', alpha=0.7)
ax.plot(best_pred, 'r-', lw=1, label='Model', alpha=0.7)
ax.set_title(f'Weekly Cases (r={corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 对数尺度
ax = fig.add_subplot(3, 3, 2)
ax.semilogy(weekly_cases + 1, 'b-', lw=1, label='Observed')
ax.semilogy(best_pred + 1, 'r-', lw=1, label='Model')
ax.set_title(f'Log Scale (R²={r2_log:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. R0
ax = fig.add_subplot(3, 3, 3)
ax.plot(R0_arr, 'g-', lw=1)
ax.axhline(y=1, color='red', ls='--')
ax.fill_between(range(n_weeks), 0, R0_arr, where=R0_arr>1, alpha=0.3, color='red')
ax.set_title('R0(t)')
ax.grid(True, alpha=0.3)

# 4. 蚊虫数量
ax = fig.add_subplot(3, 3, 4)
ax.plot(M_mosq / 1e6, 'm-', lw=1)
ax.set_title('Adult Mosquitoes (millions)')
ax.grid(True, alpha=0.3)

# 5. 蚊虫 vs 病例 (滞后)
ax = fig.add_subplot(3, 3, 5)
ax.plot(M_norm, 'g-', lw=1, label='Mosquito (norm)', alpha=0.7)
ax.plot(cases_norm, 'b-', lw=1, label='Cases (norm)', alpha=0.7)
ax.set_title(f'Mosquito vs Cases (best lag={best_lag}w, r={best_corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 温度
ax = fig.add_subplot(3, 3, 6)
ax.plot(weekly_temp, 'orange', lw=1)
ax.set_title('Temperature (°C)')
ax.grid(True, alpha=0.3)

# 7. 散点
ax = fig.add_subplot(3, 3, 7)
ax.scatter(weekly_cases, best_pred, alpha=0.5, s=20)
max_val = max(weekly_cases.max(), best_pred.max())
ax.plot([0, max_val], [0, max_val], 'k--')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Scatter')
ax.grid(True, alpha=0.3)

# 8. 蚊虫生命周期
ax = fig.add_subplot(3, 3, 8)
ax.plot(E_mosq / 1e6, 'b-', lw=0.8, label='Eggs', alpha=0.7)
ax.plot(L_mosq / 1e6, 'g-', lw=0.8, label='Larvae', alpha=0.7)
ax.plot(P_mosq / 1e6, 'y-', lw=0.8, label='Pupae', alpha=0.7)
ax.plot(M_mosq / 1e6, 'r-', lw=0.8, label='Adults', alpha=0.7)
ax.set_title('Mosquito Life Stages (millions)')
ax.legend()
ax.grid(True, alpha=0.3)

# 9. 温度依赖参数
ax = fig.add_subplot(3, 3, 9)
ax.plot(a_arr, 'b-', lw=1, label='a (bite)', alpha=0.7)
ax.plot(b_arr, 'r-', lw=1, label='b (M→H)', alpha=0.7)
ax.plot(c_arr, 'g-', lw=1, label='c (H→M)', alpha=0.7)
ax.set_title('Temperature-dependent Params')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/mosquito_human_model.png', dpi=150)
plt.close()
log("  已保存: results/figures/mosquito_human_model.png")

# 保存数据
pd.DataFrame({
    'week': range(n_weeks),
    'temperature': weekly_temp,
    'observed': weekly_cases,
    'predicted': best_pred,
    'M': M_mosq,
    'R0': R0_arr,
    'phi': phi_arr,
    'mu_m': mu_m_arr,
    'a': a_arr,
    'b': b_arr,
    'c': c_arr
}).to_csv('/root/wenmei/results/data/mosquito_human_model.csv', index=False)
log("  已保存: results/data/mosquito_human_model.csv")

# ============================================================
# 总结
# ============================================================
log("\n" + "=" * 65)
log("模型总结")
log("=" * 65)
log(f"""
【模型结构】
  蚊虫: E → L → P → M (生命周期)
  人群: S → E → I → R

【温度依赖参数 (文献公式)】
  φ(T): 产卵率
  f_e, f_l, f_p: 发育率
  μ_l, μ_p, μ_m: 死亡率
  a(T), b(T), c(T): 传播参数

【固定参数】
  δ = {DELTA:.3f} /天, γ = {GAMMA:.4f} /天

【估计参数】
  b_h = {b_h} (有效传播系数)
  imp = {imp} (输入病例)

【蚊虫-病例相关】
  最佳滞后: {best_lag} 周
  相关系数: {best_corr:.4f}

【模型性能】
  相关系数: {corr:.4f}
  R² (对数): {r2_log:.4f}
  R0 范围: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]
""")

log("\n完成!")
