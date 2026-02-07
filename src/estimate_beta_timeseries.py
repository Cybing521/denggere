#!/usr/bin/env python3
"""
估计时变传播率 β(t) 序列
用于第二阶段符号回归

方法: 给定其他参数，逐周反推 β(t) 使模型拟合观测病例
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("估计时变传播率 β(t) 序列")
log("=" * 60)

# ============================================================
# 1. 固定参数 (基于文献, 周尺度)
# ============================================================
SIGMA = 0.6       # 存活率
EPSILON = 0.7     # EIP转化率 [/周]
DELTA = 1.4       # 人潜伏期 [/周]
GAMMA = 1.0       # 人恢复率 [/周]
N_H = 14_000_000  # 人口

# 温度依赖函数 (周尺度)
def phi_T(T):
    """产卵率 [/周]"""
    return max(0.7, 28.0 * np.exp(-((T - 28) / 7)**2))

def f_l_T(T):
    """幼虫发育率 [/周]"""
    return max(0.07, 0.70 * np.exp(-((T - 27) / 9)**2))

def f_p_T(T):
    """蛹发育率 [/周]"""
    return max(0.07, 1.05 * np.exp(-((T - 27) / 9)**2))

def mu_m_T(T):
    """成蚊死亡率 [/周]"""
    return 0.28 + 0.014 * (T - 26)**2

log("""
【固定参数】(周尺度, 基于文献)
  σ = 0.6      存活率
  ε = 0.7/周   EIP转化率 (~10天)
  δ = 1.4/周   人潜伏期 (~5天)
  γ = 1.0/周   人恢复率 (~7天)

【目标】
  估计时变传播率 β(t), t = 0, 1, ..., n_weeks-1
  输出用于符号回归: β(t) = f(T(t), BI(t), ...)
""")

# ============================================================
# 2. 加载数据
# ============================================================
log("\n[1] 加载数据...")

case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()

bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
gz_bi.columns = ['year', 'month', 'bi']

df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')
df['bi'] = df['bi'].fillna(df['bi'].mean())

n_months = len(df)
n_weeks = int(n_months * 4.33)

monthly_idx = np.arange(n_months) * 4.33
weekly_idx = np.arange(n_weeks)

temp_interp = interp1d(monthly_idx, df['temperature'].values, kind='cubic', fill_value='extrapolate')
bi_interp = interp1d(monthly_idx, df['bi'].values, kind='cubic', fill_value='extrapolate')
cases_interp = interp1d(monthly_idx, df['cases'].values / 4.33, kind='linear', fill_value='extrapolate')

weekly_temp = temp_interp(weekly_idx)
weekly_bi = np.maximum(0.1, bi_interp(weekly_idx))
weekly_cases = np.maximum(0.1, cases_interp(weekly_idx))  # 避免0

bi_mean = weekly_bi.mean()
bi_normalized = weekly_bi / bi_mean

log(f"  周数: {n_weeks}")
log(f"  温度: [{weekly_temp.min():.1f}, {weekly_temp.max():.1f}]°C")
log(f"  BI: [{weekly_bi.min():.2f}, {weekly_bi.max():.2f}]")

# ============================================================
# 3. 方法1: 基于 R0 反推 β(t)
# ============================================================
log("\n[2] 方法1: 基于流行病学关系反推 β(t)...")

"""
理论基础:
  周发病数 ≈ γ * I_h
  在准稳态下: I_h ≈ (R0 - 1) * S_h / γ  (当 R0 > 1)
  
  R0 = sqrt(β² * M / (μ_m * γ * N_H))
  
  反推: β = sqrt(R0² * μ_m * γ * N_H / M)
  
  而 R0 可从病例增长率估计
"""

# 先运行一次模型获取蚊虫动态 (用之前优化的参数)
k_scale_fixed = 25.0  # 固定蚊虫参数

def run_mosquito_model(k_scale):
    """只运行蚊虫部分，获取 M(t)"""
    def mosquito_ode(y, t, temp_arr, bi_arr, n_weeks):
        L, P, M = y
        idx = min(int(t), n_weeks - 1)
        T = temp_arr[idx]
        bi_ratio = bi_arr[idx]
        
        phi = phi_T(T) * k_scale
        f_l = f_l_T(T)
        f_p = f_p_T(T)
        mu_m = mu_m_T(T)
        K_L = 1e8 * k_scale * bi_ratio
        
        dL = phi * SIGMA * M * bi_ratio - f_l * L - 0.7 * L * (1 + L/K_L)
        dP = f_l * L - f_p * P - 0.35 * P
        emergence = SIGMA * f_p * P
        dM = emergence - mu_m * M
        
        return [dL, dP, dM]
    
    L0 = 1e7 * bi_normalized[0] * k_scale
    P0 = L0 * 0.3
    M0 = P0 * 0.3
    y0 = [L0, P0, M0]
    t = np.arange(n_weeks)
    
    sol = odeint(mosquito_ode, y0, t, args=(weekly_temp, bi_normalized, n_weeks))
    return sol[:, 2]  # M(t)

M_total = run_mosquito_model(k_scale_fixed)
log(f"  蚊虫密度 M(t): [{M_total.min()/1e6:.2f}, {M_total.max()/1e6:.2f}] M")

# 从病例增长率估计 R_eff，再反推 β
log("\n  从病例动态反推 β(t)...")

# 平滑病例数据
from scipy.ndimage import gaussian_filter1d
cases_smooth = gaussian_filter1d(weekly_cases, sigma=2)

# 估计有效再生数 R_eff (Wallinga-Teunis 方法简化版)
# R_eff ≈ cases(t) / cases(t-1) * (1/γ)，考虑代际间隔
generation_time = 2  # 约2周代际间隔
R_eff = np.ones(n_weeks)
for t in range(generation_time, n_weeks):
    if cases_smooth[t - generation_time] > 0.1:
        R_eff[t] = cases_smooth[t] / cases_smooth[t - generation_time]
    else:
        R_eff[t] = 0.1

# 限制 R_eff 范围
R_eff = np.clip(R_eff, 0.01, 10)
R_eff = gaussian_filter1d(R_eff, sigma=1)

# 从 R_eff 反推 β(t)
# R0 = sqrt(β² * M / (μ_m * γ * N_H))
# β = sqrt(R0² * μ_m * γ * N_H / M)
beta_estimated = np.zeros(n_weeks)
for t in range(n_weeks):
    T = weekly_temp[t]
    M = max(M_total[t], 1)
    mu_m = mu_m_T(T)
    
    R0_sq = R_eff[t] ** 2
    beta_sq = R0_sq * mu_m * GAMMA * N_H / M
    beta_estimated[t] = np.sqrt(max(beta_sq, 1e-10))

# 限制 β 范围 (合理的传播率)
beta_estimated = np.clip(beta_estimated, 0.001, 10)

log(f"  β(t) 范围: [{beta_estimated.min():.4f}, {beta_estimated.max():.4f}]")

# ============================================================
# 4. 方法2: 逐周优化 β(t)
# ============================================================
log("\n[3] 方法2: 逐周优化 β(t)...")

def full_model_with_beta_series(beta_series, k_scale, imp):
    """给定 β(t) 序列，运行完整模型"""
    def ode_system(y, t, beta_arr, temp_arr, bi_arr, n_weeks):
        L, P, S_m, E_m, I_m, S_h, E_h, I_h = y
        
        idx = min(int(t), n_weeks - 1)
        T = temp_arr[idx]
        bi_ratio = bi_arr[idx]
        beta = beta_arr[idx]
        
        phi = phi_T(T) * k_scale
        f_l = f_l_T(T)
        f_p = f_p_T(T)
        mu_m = mu_m_T(T)
        
        K_L = 1e8 * k_scale * bi_ratio
        M = max(S_m + E_m + I_m, 1)
        
        dL = phi * SIGMA * M * bi_ratio - f_l * L - 0.7 * L * (1 + L/K_L)
        dP = f_l * L - f_p * P - 0.35 * P
        
        emergence = SIGMA * f_p * P
        lambda_m = beta * (I_h + imp) / N_H
        
        dS_m = emergence - lambda_m * S_m - mu_m * S_m
        dE_m = lambda_m * S_m - EPSILON * E_m - mu_m * E_m
        dI_m = EPSILON * E_m - mu_m * I_m
        
        lambda_h = beta * I_m / N_H
        dS_h = -lambda_h * S_h
        dE_h = lambda_h * S_h - DELTA * E_h
        dI_h = DELTA * E_h - GAMMA * I_h
        
        return [dL, dP, dS_m, dE_m, dI_m, dS_h, dE_h, dI_h]
    
    L0 = 1e7 * bi_normalized[0] * k_scale
    P0 = L0 * 0.3
    M0 = P0 * 0.3
    I_h0 = max(weekly_cases[0], 1) / GAMMA
    E_h0 = I_h0 * 0.5
    S_h0 = N_H - E_h0 - I_h0
    
    y0 = [L0, P0, M0*0.95, M0*0.03, M0*0.02, S_h0, E_h0, I_h0]
    t = np.arange(n_weeks)
    
    sol = odeint(ode_system, y0, t, 
                 args=(beta_series, weekly_temp, bi_normalized, n_weeks),
                 rtol=1e-5, atol=1e-8)
    
    I_h = sol[:, 7]
    pred = GAMMA * I_h
    return sol, np.maximum(pred, 0)

# 使用方法1的估计作为初始值，优化整个 β 序列
imp_fixed = 50  # 固定输入病例

def objective_beta_series(beta_flat):
    """优化整个 β(t) 序列"""
    beta_series = np.maximum(beta_flat, 0.001)
    _, pred = full_model_with_beta_series(beta_series, k_scale_fixed, imp_fixed)
    mse = np.mean((np.log1p(weekly_cases) - np.log1p(pred))**2)
    
    # 平滑性惩罚 (β不应该剧烈变化)
    smoothness = np.mean(np.diff(beta_series)**2)
    
    return mse + 0.01 * smoothness

# 分段优化 (太多参数一起优化很慢)
log("  分段优化 β(t)...")
beta_optimized = beta_estimated.copy()

# 使用滑动窗口优化
window_size = 10
for start in range(0, n_weeks - window_size, window_size // 2):
    end = min(start + window_size, n_weeks)
    
    def local_objective(beta_window):
        beta_test = beta_optimized.copy()
        beta_test[start:end] = np.maximum(beta_window, 0.001)
        _, pred = full_model_with_beta_series(beta_test, k_scale_fixed, imp_fixed)
        
        # 只看这个窗口的拟合
        mse = np.mean((np.log1p(weekly_cases[start:end]) - np.log1p(pred[start:end]))**2)
        return mse
    
    result = minimize(local_objective, beta_optimized[start:end],
                     method='L-BFGS-B',
                     bounds=[(0.001, 10)] * (end - start),
                     options={'maxiter': 50})
    
    beta_optimized[start:end] = np.maximum(result.x, 0.001)
    
    if (start // (window_size // 2)) % 10 == 0:
        log(f"    窗口 {start}-{end}: loss={result.fun:.4f}")

# 平滑最终结果
beta_optimized = gaussian_filter1d(beta_optimized, sigma=1)
beta_optimized = np.clip(beta_optimized, 0.001, 10)

log(f"  优化后 β(t): [{beta_optimized.min():.4f}, {beta_optimized.max():.4f}]")

# ============================================================
# 5. 评估
# ============================================================
log("\n[4] 评估...")

sol, weekly_pred = full_model_with_beta_series(beta_optimized, k_scale_fixed, imp_fixed)
L, P, S_m, E_m, I_m, S_h, E_h, I_h = sol.T
M_model = S_m + E_m + I_m

corr, pval = pearsonr(weekly_cases, weekly_pred)
r2_log = r2_score(np.log1p(weekly_cases), np.log1p(weekly_pred))

log(f"  Pearson r = {corr:.4f}")
log(f"  R² (log) = {r2_log:.4f}")

# β 与环境变量的相关性
corr_beta_T, _ = pearsonr(beta_optimized, weekly_temp)
corr_beta_BI, _ = pearsonr(beta_optimized, weekly_bi)
log(f"  β-温度相关: {corr_beta_T:.4f}")
log(f"  β-BI相关: {corr_beta_BI:.4f}")

# ============================================================
# 6. 可视化
# ============================================================
log("\n[5] 生成图形...")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

# 1. 病例拟合
ax = axes[0, 0]
ax.plot(weekly_cases, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax.plot(weekly_pred, 'r-', lw=1.5, label='Model', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Cases')
ax.set_title(f'Cases Fit (r={corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. β(t) 时间序列
ax = axes[0, 1]
ax.plot(beta_optimized, 'g-', lw=1.5)
ax.set_xlabel('Week')
ax.set_ylabel('β(t)')
ax.set_title('Time-varying Transmission Rate β(t)')
ax.grid(True, alpha=0.3)

# 3. β vs 温度
ax = axes[0, 2]
ax.scatter(weekly_temp, beta_optimized, alpha=0.5, s=30, c=range(n_weeks), cmap='viridis')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('β(t)')
ax.set_title(f'β vs Temperature (r={corr_beta_T:.3f})')
ax.grid(True, alpha=0.3)

# 4. β vs BI
ax = axes[1, 0]
ax.scatter(weekly_bi, beta_optimized, alpha=0.5, s=30, c=range(n_weeks), cmap='viridis')
ax.set_xlabel('BI')
ax.set_ylabel('β(t)')
ax.set_title(f'β vs BI (r={corr_beta_BI:.3f})')
ax.grid(True, alpha=0.3)

# 5. β 和温度时间序列对比
ax = axes[1, 1]
ax2 = ax.twinx()
ax.plot(weekly_temp, 'orange', lw=1.5, label='Temperature')
ax2.plot(beta_optimized, 'g-', lw=1.5, label='β(t)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Temperature (°C)', color='orange')
ax2.set_ylabel('β(t)', color='green')
ax.set_title('Temperature and β(t)')
ax.grid(True, alpha=0.3)

# 6. β 和病例
ax = axes[1, 2]
ax2 = ax.twinx()
ax.plot(weekly_cases, 'b-', lw=1, label='Cases', alpha=0.5)
ax2.plot(beta_optimized, 'g-', lw=1.5, label='β(t)')
ax.set_xlabel('Week')
ax.set_ylabel('Cases', color='blue')
ax2.set_ylabel('β(t)', color='green')
ax.set_title('Cases and β(t)')
ax.grid(True, alpha=0.3)

# 7. 蚊虫密度
ax = axes[2, 0]
ax.plot(M_model / 1e6, 'g-', lw=1.5)
ax.set_xlabel('Week')
ax.set_ylabel('Mosquitoes (M)')
ax.set_title('Adult Mosquito Population')
ax.grid(True, alpha=0.3)

# 8. R0(t)
R0_arr = np.sqrt(beta_optimized**2 * M_model / (np.array([mu_m_T(T) for T in weekly_temp]) * GAMMA * N_H + 1e-10))
ax = axes[2, 1]
ax.plot(R0_arr, 'purple', lw=1.5)
ax.axhline(y=1, color='red', ls='--', lw=2)
ax.fill_between(range(n_weeks), 0, R0_arr, where=R0_arr > 1, alpha=0.3, color='red')
ax.set_xlabel('Week')
ax.set_ylabel('R0')
ax.set_title('Reproduction Number')
ax.grid(True, alpha=0.3)

# 9. 散点图
ax = axes[2, 2]
ax.scatter(weekly_cases, weekly_pred, alpha=0.5, s=30)
max_val = max(weekly_cases.max(), weekly_pred.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Scatter Plot')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/beta_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
log("  已保存: results/figures/beta_timeseries.png")

# ============================================================
# 7. 保存 β(t) 序列用于符号回归
# ============================================================
log("\n[6] 保存数据...")

result_df = pd.DataFrame({
    'week': range(n_weeks),
    'year': [2015 + i // 52 for i in range(n_weeks)],
    'week_of_year': [i % 52 for i in range(n_weeks)],
    'temperature': weekly_temp,
    'bi': weekly_bi,
    'bi_normalized': bi_normalized,
    'observed_cases': weekly_cases,
    'predicted_cases': weekly_pred,
    'beta': beta_optimized,  # ★ 时变传播率，用于符号回归
    'M_mosquito': M_model,
    'R0': R0_arr,
    'I_h': I_h
})
result_df.to_csv('/root/wenmei/results/data/beta_timeseries.csv', index=False)
log("  已保存: results/data/beta_timeseries.csv")

# 符号回归输入数据 (简化版)
sr_df = pd.DataFrame({
    'beta': beta_optimized,       # 目标变量 y
    'temperature': weekly_temp,   # 特征 x1
    'bi': weekly_bi,              # 特征 x2
    'bi_normalized': bi_normalized,
    'mosquito_density': M_model / 1e6,  # 特征 x3 (百万)
})
sr_df.to_csv('/root/wenmei/results/data/symbolic_regression_input.csv', index=False)
log("  已保存: results/data/symbolic_regression_input.csv")
log("  ↑ 此文件用于第二阶段符号回归")

# ============================================================
# 总结
# ============================================================
log("\n" + "=" * 60)
log("时变 β(t) 估计总结")
log("=" * 60)
log(f"""
【方法】
  1. 运行蚊虫动态模型获取 M(t)
  2. 从病例增长率估计 R_eff(t)
  3. 反推初始 β(t) = sqrt(R0² * μ_m * γ * N_H / M)
  4. 分段优化 β(t) 使模型拟合观测病例

【结果】
  β(t) 范围: [{beta_optimized.min():.4f}, {beta_optimized.max():.4f}]
  模型拟合: r = {corr:.4f}, R² = {r2_log:.4f}

【β(t) 与环境变量相关性】
  β-温度: r = {corr_beta_T:.4f}
  β-BI:   r = {corr_beta_BI:.4f}

【符号回归输入】
  文件: results/data/symbolic_regression_input.csv
  目标: β = f(temperature, bi, mosquito_density, ...)
""")
log("完成! 可以进行第二阶段符号回归了。")
