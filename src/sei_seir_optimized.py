#!/usr/bin/env python3
"""
优化版 SEI-SEIR 模型
改进:
1. 更精细的参数搜索 (差分进化 + 局部优化)
2. 季节性承载力
3. 改进初始条件估计
4. 增强BI校正机制
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("优化版 SEI-SEIR 动力学模型")
print("=" * 60)

# ============================================================
# 1. 温度依赖参数 (改进版)
# ============================================================
def phi_T(T, phi_max=4.0):
    """产卵率 - 更平滑"""
    if T < 10 or T > 40:
        return 0.01
    return max(0.1, phi_max * np.exp(-((T - 28) / 7)**2))

def f_e_T(T):
    """卵孵化率"""
    if T < 12 or T > 38:
        return 0.005
    return max(0.01, 0.18 * np.exp(-((T - 27) / 9)**2))

def f_l_T(T):
    """幼虫发育率 (控制滞后)"""
    if T < 12 or T > 38:
        return 0.005
    return max(0.01, 0.10 * np.exp(-((T - 27) / 9)**2))

def f_p_T(T):
    """蛹发育率"""
    if T < 12 or T > 38:
        return 0.005
    return max(0.01, 0.15 * np.exp(-((T - 27) / 9)**2))

def mu_l_T(T):
    """幼虫死亡率"""
    if T < 15:
        return 0.3
    elif T > 35:
        return 0.25
    return max(0.05, 0.15 - 0.006 * (T - 15))

def mu_m_T(T):
    """成蚊死亡率 - U形曲线"""
    optimal = 26
    mu_min = 0.04
    return mu_min + 0.002 * (T - optimal)**2

def a_T(T):
    """叮咬率"""
    if T < 14 or T > 38:
        return 0.05
    return max(0.1, 0.5 * np.exp(-((T - 28) / 8)**2))

def b_hv_T(T):
    """人→蚊传播概率"""
    if T < 14 or T > 35:
        return 0.01
    return 0.4 * np.exp(-((T - 27) / 6)**2)

def b_vh_T(T):
    """蚊→人传播概率"""
    if T < 14 or T > 35:
        return 0.01
    return 0.45 * np.exp(-((T - 26) / 6)**2)

def eip_rate_T(T):
    """外潜伏期转化率 (温度依赖)"""
    if T < 18:
        return 0.05  # 慢
    elif T > 32:
        return 0.2   # 快
    return 0.08 + 0.01 * (T - 20)

# 固定参数
SIGMA_E = 0.6    # 卵存活率
SIGMA_P = 0.7    # 蛹存活率  
DELTA = 1/5      # 人潜伏期
GAMMA = 1/7      # 人恢复率
N_H = 14_000_000

# ============================================================
# 2. 加载数据
# ============================================================
print("\n[1] 加载数据...")

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
weekly_cases = np.maximum(0, cases_interp(weekly_idx))

# BI 标准化
bi_mean = weekly_bi.mean()
bi_std = weekly_bi.std()
bi_normalized = weekly_bi / bi_mean

# 年份信息
weeks_per_year = int(52)
n_years = 5
year_starts = [i * weeks_per_year for i in range(n_years)]

print(f"  周数据: {n_weeks}周")
print(f"  总病例: {weekly_cases.sum():.0f}")
print(f"  BI范围: [{weekly_bi.min():.2f}, {weekly_bi.max():.2f}]")
print(f"  温度范围: [{weekly_temp.min():.1f}, {weekly_temp.max():.1f}]°C")

# ============================================================
# 3. 改进的ODE系统
# ============================================================
def ode_system(y, t, params, temp_arr, bi_arr, n_weeks):
    """
    改进的SEI-SEIR系统
    状态: L, P, S_m, E_m, I_m, E_h, I_h
    """
    L, P, S_m, E_m, I_m, E_h, I_h = y
    
    idx = min(int(t / 7), n_weeks - 1)
    T = temp_arr[idx]
    bi_ratio = bi_arr[idx]
    
    # 解包参数
    k_scale, b_scale, imp_rate, K_base = params
    
    # 温度依赖率
    phi = phi_T(T) * k_scale
    f_l = f_l_T(T)
    f_p = f_p_T(T)
    mu_l = mu_l_T(T)
    mu_m = mu_m_T(T)
    a = a_T(T)
    b_hv = b_hv_T(T) * b_scale
    b_vh = b_vh_T(T) * b_scale
    eip = eip_rate_T(T)
    
    # 季节性承载力 (BI驱动)
    K_L = K_base * 1e7 * bi_ratio
    
    M_total = max(S_m + E_m + I_m, 1)
    
    # === 幼虫动态 (BI校正) ===
    egg_laying = phi * SIGMA_E * M_total * bi_ratio  # BI 增强产卵
    development = f_l * L
    mortality = mu_l * L * (1 + L / K_L)  # 密度依赖
    dL = egg_laying - development - mortality
    
    # === 蛹 ===
    dP = f_l * L - f_p * P - 0.05 * P
    
    # === 成蚊 SEI ===
    emergence = SIGMA_P * f_p * P
    
    # 感染力 (考虑输入病例)
    I_eff = I_h + imp_rate
    lambda_m = a * b_hv * I_eff / N_H
    
    dS_m = emergence - lambda_m * S_m - mu_m * S_m
    dE_m = lambda_m * S_m - eip * E_m - mu_m * E_m
    dI_m = eip * E_m - mu_m * I_m
    
    # === 人群 SEIR ===
    S_h = N_H - E_h - I_h
    lambda_h = a * b_vh * I_m / N_H
    
    dE_h = lambda_h * S_h - DELTA * E_h
    dI_h = DELTA * E_h - GAMMA * I_h
    
    return [dL, dP, dS_m, dE_m, dI_m, dE_h, dI_h]


def run_model(params, return_full=False):
    """运行模型并返回预测"""
    k_scale, b_scale, imp_rate, K_base = params
    
    # 初始条件 (基于第一周数据估计)
    L0 = K_base * 1e6 * bi_normalized[0]
    P0 = L0 * 0.3
    M0 = P0 * 0.5
    
    # 根据第一周病例估计初始人群状态
    case0 = max(weekly_cases[0], 1)
    I_h0 = case0 / GAMMA  # 稳态近似
    E_h0 = I_h0 * DELTA / GAMMA
    
    y0 = [L0, P0, M0*0.95, M0*0.03, M0*0.02, E_h0, I_h0]
    
    t = np.arange(n_weeks) * 7
    
    try:
        sol = odeint(ode_system, y0, t,
                     args=(params, weekly_temp, bi_normalized, n_weeks),
                     rtol=1e-6, atol=1e-9, mxstep=5000)
        
        I_h = sol[:, 6]
        pred = GAMMA * I_h * 7  # 周发病数
        pred = np.maximum(pred, 0)
        
        if return_full:
            return sol, pred
        return pred
    except:
        if return_full:
            return None, None
        return None


def objective(params):
    """综合目标函数"""
    pred = run_model(params)
    
    if pred is None or np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        return 1e10
    
    # 1. 对数MSE (主要)
    log_obs = np.log1p(weekly_cases)
    log_pred = np.log1p(pred)
    mse_log = np.mean((log_obs - log_pred)**2)
    
    # 2. 相关性奖励
    if np.std(pred) > 1:
        corr, _ = pearsonr(weekly_cases, pred)
        if not np.isnan(corr):
            corr_bonus = -0.5 * max(0, corr)
        else:
            corr_bonus = 0
    else:
        corr_bonus = 1.0  # 惩罚无变化
    
    # 3. 峰值匹配
    obs_max_idx = np.argmax(weekly_cases)
    pred_max_idx = np.argmax(pred)
    peak_diff = abs(obs_max_idx - pred_max_idx) / n_weeks
    peak_penalty = 0.3 * peak_diff
    
    # 4. 季节性匹配 (每年高峰期)
    seasonal_penalty = 0
    for yr in range(n_years):
        start = yr * weeks_per_year
        end = min((yr + 1) * weeks_per_year, n_weeks)
        if end > start + 10:
            obs_yr = weekly_cases[start:end]
            pred_yr = pred[start:end]
            if obs_yr.max() > 10 and pred_yr.max() > 1:
                obs_peak = np.argmax(obs_yr)
                pred_peak = np.argmax(pred_yr)
                seasonal_penalty += 0.05 * abs(obs_peak - pred_peak) / (end - start)
    
    return mse_log + corr_bonus + peak_penalty + seasonal_penalty

# ============================================================
# 4. 两阶段优化 (带进度显示)
# ============================================================
print("\n[2] 第一阶段: 全局搜索...")

bounds = [
    (0.1, 20.0),    # k_scale
    (0.01, 5.0),    # b_scale
    (0.0, 50.0),    # imp_rate
    (0.1, 10.0)     # K_base
]

# 进度回调
iteration_count = [0]
def callback(xk, convergence=None):
    iteration_count[0] += 1
    if iteration_count[0] % 10 == 0:
        loss = objective(xk)
        print(f"  迭代 {iteration_count[0]}: loss={loss:.4f}, params=[{xk[0]:.2f},{xk[1]:.3f},{xk[2]:.1f},{xk[3]:.2f}]", flush=True)

# 阶段1: 差分进化 (减少迭代)
result1 = differential_evolution(
    objective, bounds, 
    seed=42, 
    maxiter=60,  # 减少迭代
    tol=1e-5,
    workers=1,
    mutation=(0.5, 1.0),
    recombination=0.7,
    polish=False,
    updating='immediate',
    popsize=10,  # 减少种群
    callback=callback
)

print(f"  阶段1 Loss: {result1.fun:.4f}")
print(f"  参数: k={result1.x[0]:.3f}, b={result1.x[1]:.3f}, imp={result1.x[2]:.2f}, K={result1.x[3]:.3f}")

print("\n[3] 第二阶段: 局部精细化...", flush=True)

# 阶段2: 局部优化 (简化)
best_result = result1
best_loss = result1.fun

for i in range(3):  # 减少重启次数
    x0 = result1.x * (1 + 0.05 * np.random.randn(4))
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    
    try:
        result_local = minimize(
            objective, x0,
            method='Nelder-Mead',
            options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-5}
        )
        
        if result_local.fun < best_loss:
            best_loss = result_local.fun
            best_result = result_local
            print(f"  迭代{i+1}: 改进! Loss = {best_loss:.4f}", flush=True)
    except:
        pass

# 最终参数
k_scale, b_scale, imp_rate, K_base = best_result.x
print(f"\n  最终参数:")
print(f"    k_scale = {k_scale:.4f}")
print(f"    b_scale = {b_scale:.4f}")
print(f"    imp_rate = {imp_rate:.2f} /周")
print(f"    K_base = {K_base:.4f}")
print(f"    最终 Loss = {best_loss:.4f}")

# ============================================================
# 5. 运行最终模型
# ============================================================
print("\n[4] 运行最终模型...")

sol, weekly_pred = run_model(best_result.x, return_full=True)
L, P, S_m, E_m, I_m, E_h, I_h = sol.T
M_total = S_m + E_m + I_m

# ============================================================
# 6. 全面评估
# ============================================================
print("\n[5] 模型评估...")

# 基础指标
corr_cases, pval = pearsonr(weekly_cases, weekly_pred)
spearman_r, _ = spearmanr(weekly_cases, weekly_pred)
r2_log = r2_score(np.log1p(weekly_cases), np.log1p(weekly_pred))
rmse = np.sqrt(mean_squared_error(weekly_cases, weekly_pred))
rmse_log = np.sqrt(mean_squared_error(np.log1p(weekly_cases), np.log1p(weekly_pred)))

# BI-幼虫相关
L_norm = L / (L.mean() + 1e-10)
corr_bi_L, _ = pearsonr(bi_normalized, L_norm)

# 分年评估
print("\n  分年相关性:")
for yr in range(n_years):
    start = yr * weeks_per_year
    end = min((yr + 1) * weeks_per_year, n_weeks)
    if end > start:
        obs_yr = weekly_cases[start:end]
        pred_yr = weekly_pred[start:end]
        if np.std(obs_yr) > 0 and np.std(pred_yr) > 0:
            r_yr, _ = pearsonr(obs_yr, pred_yr)
            print(f"    {2015+yr}: r = {r_yr:.4f}")

# R0 计算
R0_arr = []
for i in range(n_weeks):
    T = weekly_temp[i]
    M = M_total[i]
    a = a_T(T)
    b_vh = b_vh_T(T) * b_scale
    b_hv = b_hv_T(T) * b_scale
    mu_m = mu_m_T(T)
    eip = eip_rate_T(T)
    
    # R0 = √(a² * b_vh * b_hv * M * eip / (mu_m * (eip + mu_m) * GAMMA * N_H))
    numerator = a**2 * b_vh * b_hv * M * eip
    denominator = mu_m * (eip + mu_m) * GAMMA * N_H + 1e-10
    R0 = np.sqrt(numerator / denominator)
    R0_arr.append(R0)
R0_arr = np.array(R0_arr)

print(f"\n  总体指标:")
print(f"    Pearson r = {corr_cases:.4f}")
print(f"    Spearman ρ = {spearman_r:.4f}")
print(f"    R² (log) = {r2_log:.4f}")
print(f"    RMSE = {rmse:.2f}")
print(f"    RMSE (log) = {rmse_log:.4f}")
print(f"    BI-L相关 = {corr_bi_L:.4f}")
print(f"    R0: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]")
print(f"    R0>1 周数 = {(R0_arr > 1).sum()}/{n_weeks}")

# ============================================================
# 7. 可视化
# ============================================================
print("\n[6] 生成图形...")

fig, axes = plt.subplots(4, 3, figsize=(18, 20))

# 1. 主图: 病例拟合
ax = axes[0, 0]
ax.plot(weekly_cases, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax.plot(weekly_pred, 'r-', lw=1.5, label='Predicted', alpha=0.8)
for yr in range(1, n_years):
    ax.axvline(x=yr*weeks_per_year, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Week')
ax.set_ylabel('Weekly Cases')
ax.set_title(f'Dengue Cases Fit (r={corr_cases:.3f}, p={pval:.2e})')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 2. 对数尺度
ax = axes[0, 1]
ax.semilogy(weekly_cases + 1, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax.semilogy(weekly_pred + 1, 'r-', lw=1.5, label='Predicted', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Cases (log scale)')
ax.set_title(f'Log Scale (R²={r2_log:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 散点图
ax = axes[0, 2]
ax.scatter(weekly_cases, weekly_pred, alpha=0.5, s=40, c=range(n_weeks), cmap='viridis')
max_val = max(weekly_cases.max(), weekly_pred.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2, label='1:1')
ax.set_xlabel('Observed Cases')
ax.set_ylabel('Predicted Cases')
ax.set_title('Scatter Plot')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. R0
ax = axes[1, 0]
ax.plot(R0_arr, 'g-', lw=1.5)
ax.axhline(y=1, color='red', ls='--', lw=2, label='R0=1')
ax.fill_between(range(n_weeks), 0, R0_arr, where=R0_arr > 1, alpha=0.3, color='red', label='R0>1')
for yr in range(1, n_years):
    ax.axvline(x=yr*weeks_per_year, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Week')
ax.set_ylabel('R0')
ax.set_title(f'Reproduction Number (max={R0_arr.max():.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. BI vs 幼虫
ax = axes[1, 1]
ax.plot(bi_normalized, 'g-', lw=1.5, label='BI (normalized)', alpha=0.8)
ax.plot(L_norm, 'b-', lw=1.5, label='L (model, normalized)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Normalized Value')
ax.set_title(f'BI vs Larvae Density (r={corr_bi_L:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 水生阶段
ax = axes[1, 2]
ax.plot(L / 1e6, 'b-', lw=1, label='L (larvae)', alpha=0.8)
ax.plot(P / 1e6, 'orange', lw=1, label='P (pupae)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Population (millions)')
ax.set_title('Aquatic Stages')
ax.legend()
ax.grid(True, alpha=0.3)

# 7. 成蚊动态
ax = axes[2, 0]
ax.plot(M_total / 1e6, 'g-', lw=1.5, label='Total Adults', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Mosquitoes (millions)')
ax.set_title('Adult Mosquito Population')
ax.legend()
ax.grid(True, alpha=0.3)

# 8. 感染蚊
ax = axes[2, 1]
ax.plot(I_m, 'r-', lw=1.5)
ax.set_xlabel('Week')
ax.set_ylabel('Infected Mosquitoes')
ax.set_title('I_m Dynamics')
ax.grid(True, alpha=0.3)

# 9. 人群动态
ax = axes[2, 2]
ax.plot(E_h, 'y-', lw=1.5, label='E_h (exposed)', alpha=0.8)
ax.plot(I_h, 'r-', lw=1.5, label='I_h (infected)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Human Population')
ax.set_title('Human E & I Compartments')
ax.legend()
ax.grid(True, alpha=0.3)

# 10. 温度影响
ax = axes[3, 0]
ax.plot(weekly_temp, 'orange', lw=1.5, label='Temperature')
ax2 = ax.twinx()
ax2.plot(weekly_cases, 'b-', lw=1, alpha=0.5, label='Cases')
ax.set_xlabel('Week')
ax.set_ylabel('Temperature (°C)', color='orange')
ax2.set_ylabel('Cases', color='blue')
ax.set_title('Temperature & Cases')
ax.grid(True, alpha=0.3)

# 11. 残差
ax = axes[3, 1]
residuals = weekly_cases - weekly_pred
ax.bar(range(n_weeks), residuals, alpha=0.6, width=1)
ax.axhline(y=0, color='black', lw=1)
ax.set_xlabel('Week')
ax.set_ylabel('Residual (Obs - Pred)')
ax.set_title('Residuals')
ax.grid(True, alpha=0.3)

# 12. 分年对比
ax = axes[3, 2]
colors = plt.cm.Set1(np.linspace(0, 1, n_years))
for yr in range(n_years):
    start = yr * weeks_per_year
    end = min((yr + 1) * weeks_per_year, n_weeks)
    week_in_year = np.arange(end - start)
    ax.plot(week_in_year, weekly_cases[start:end], '-', color=colors[yr], 
            alpha=0.5, lw=1, label=f'{2015+yr} Obs')
    ax.plot(week_in_year, weekly_pred[start:end], '--', color=colors[yr], 
            alpha=0.8, lw=1.5)
ax.set_xlabel('Week of Year')
ax.set_ylabel('Cases')
ax.set_title('Year-by-Year Comparison (solid=obs, dashed=pred)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/sei_seir_optimized.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: results/figures/sei_seir_optimized.png")

# 保存结果
result_df = pd.DataFrame({
    'week': range(n_weeks),
    'year': [2015 + i // weeks_per_year for i in range(n_weeks)],
    'week_of_year': [i % weeks_per_year for i in range(n_weeks)],
    'temperature': weekly_temp,
    'bi': weekly_bi,
    'bi_normalized': bi_normalized,
    'observed_cases': weekly_cases,
    'predicted_cases': weekly_pred,
    'L': L,
    'P': P,
    'S_m': S_m,
    'E_m': E_m,
    'I_m': I_m,
    'M_total': M_total,
    'E_h': E_h,
    'I_h': I_h,
    'R0': R0_arr
})
result_df.to_csv('/root/wenmei/results/data/sei_seir_optimized.csv', index=False)
print("  已保存: results/data/sei_seir_optimized.csv")

# ============================================================
# 8. 总结报告
# ============================================================
print("\n" + "=" * 60)
print("优化版模型总结")
print("=" * 60)
print(f"""
【模型结构】
  蚊虫水生期: L (幼虫) → P (蛹)
  蚊虫成虫: S_m → E_m → I_m
  人群: S_h → E_h → I_h → R_h

【BI校正机制】
  1. 产卵率: φ × bi_ratio (BI高 → 产卵多)
  2. 承载力: K_L = K_base × bi_ratio (BI高 → 承载力大)
  3. 相关性: BI-幼虫 r = {corr_bi_L:.4f}

【优化参数】
  k_scale = {k_scale:.4f} (产卵/承载力缩放)
  b_scale = {b_scale:.4f} (传播效率缩放)
  imp_rate = {imp_rate:.2f} /周 (输入病例)
  K_base = {K_base:.4f} (基础承载力因子)

【模型性能】
  Pearson r = {corr_cases:.4f} (p = {pval:.2e})
  Spearman ρ = {spearman_r:.4f}
  R² (log) = {r2_log:.4f}
  RMSE = {rmse:.2f}
  RMSE (log) = {rmse_log:.4f}

【基本再生数】
  R0 范围: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]
  R0>1 周数: {(R0_arr > 1).sum()}/{n_weeks} ({100*(R0_arr>1).sum()/n_weeks:.1f}%)

【内在滞后】
  温度 → 幼虫(L) → 蛹(P) → 成蚊(M) → 感染蚊(I_m) → 人感染(I_h)
  发育率控制自然滞后 ≈ 2-4 周
""")
print("优化完成!")
