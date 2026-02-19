#!/usr/bin/env python3
"""
快速版 SEI-SEIR 模型 (用BI校正幼虫)
"""
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("完整蚊虫-人群 SEI-SEIR 动力学模型 (快速版)")
print("=" * 60)

# ============================================================
# 1. 温度依赖参数
# ============================================================
def phi_T(T):
    """产卵率"""
    return max(0.1, 3.5 * np.exp(-((T - 28) / 8)**2))

def f_e_T(T):
    """卵孵化率"""
    return max(0.01, 0.15 * np.exp(-((T - 28) / 10)**2))

def f_l_T(T):
    """幼虫发育率"""
    return max(0.01, 0.08 * np.exp(-((T - 27) / 10)**2))

def f_p_T(T):
    """蛹发育率"""  
    return max(0.01, 0.12 * np.exp(-((T - 27) / 10)**2))

def mu_m_T(T):
    """成蚊死亡率"""
    return max(0.03, 0.15 - 0.004 * T + 0.0001 * T**2)

def b_hv_T(T):
    """人→蚊传播"""
    if T < 14 or T > 35:
        return 0.001
    return 0.3 * np.exp(-((T - 27) / 6)**2)

def b_vh_T(T):
    """蚊→人传播"""
    if T < 14 or T > 35:
        return 0.001
    return 0.35 * np.exp(-((T - 26) / 6)**2)

SIGMA = 0.5
EPSILON = 0.1    # EIP 转化率
DELTA = 1/5
GAMMA = 1/6
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

bi_mean = weekly_bi.mean()
bi_normalized = weekly_bi / bi_mean

print(f"  周数据: {n_weeks}周, 总病例: {weekly_cases.sum():.0f}")

# ============================================================
# 3. 简化ODE
# ============================================================
def ode_system(y, t, params, temp_arr, bi_arr, n_weeks):
    """简化的SEI-SEIR"""
    L, P, S_m, E_m, I_m, E_h, I_h = y
    
    idx = min(int(t / 7), n_weeks - 1)
    T = temp_arr[idx]
    bi_ratio = bi_arr[idx]
    
    k_scale, b_scale, imp = params
    
    phi = phi_T(T) * k_scale
    f_l = f_l_T(T)
    f_p = f_p_T(T)
    mu_m = mu_m_T(T)
    b_hv = b_hv_T(T) * b_scale
    b_vh = b_vh_T(T) * b_scale
    
    K_L = 1e8 * k_scale
    M_total = S_m + E_m + I_m
    
    # 幼虫 (BI校正)
    dL = phi * SIGMA * M_total * bi_ratio - f_l * L - 0.1 * L * (1 + L/K_L)
    
    # 蛹
    dP = f_l * L - f_p * P - 0.05 * P
    
    # 成蚊 SEI
    emergence = SIGMA * f_p * P
    inf_m = b_hv * (I_h + imp) / N_H
    
    dS_m = emergence - inf_m * S_m - mu_m * S_m
    dE_m = inf_m * S_m - EPSILON * E_m - mu_m * E_m
    dI_m = EPSILON * E_m - mu_m * I_m
    
    # 人群
    inf_h = b_vh * I_m / N_H
    S_h = N_H - E_h - I_h
    
    dE_h = inf_h * S_h - DELTA * E_h
    dI_h = DELTA * E_h - GAMMA * I_h
    
    return [dL, dP, dS_m, dE_m, dI_m, dE_h, dI_h]


def objective(params):
    k_scale, b_scale, imp = params
    
    L0 = 1e7 * bi_normalized[0]
    P0 = 5e6
    M0 = 1e6
    y0 = [L0, P0, M0*0.97, M0*0.02, M0*0.01, 200, 100]
    
    t = np.arange(n_weeks) * 7
    
    try:
        sol = odeint(ode_system, y0, t, 
                     args=(params, weekly_temp, bi_normalized, n_weeks),
                     rtol=1e-5, atol=1e-8)
        
        I_h = sol[:, 6]
        pred = GAMMA * I_h * 7
        pred = np.maximum(pred, 0)
        
        # 对数MSE
        mse = np.mean((np.log1p(weekly_cases) - np.log1p(pred))**2)
        
        # 相关性
        if np.std(pred) > 1:
            corr, _ = pearsonr(weekly_cases, pred)
            if not np.isnan(corr) and corr > 0:
                mse -= 0.3 * corr
        
        # BI-L相关
        L = sol[:, 0]
        if L.std() > 1:
            corr_L, _ = pearsonr(bi_normalized, L / L.mean())
            if corr_L < 0.2:
                mse += 0.5
        
        return mse
    except:
        return 1e10

# ============================================================
# 4. 优化
# ============================================================
print("\n[2] 参数估计...")

from scipy.optimize import differential_evolution

bounds = [(0.1, 10), (0.01, 10), (0, 20)]

print("  优化中 (快速模式)...")
result = differential_evolution(objective, bounds, seed=42, maxiter=80, 
                                 tol=1e-5, workers=1, polish=True,
                                 mutation=(0.5, 1), recombination=0.7)

k_scale, b_scale, imp = result.x
print(f"  k_scale = {k_scale:.4f}")
print(f"  b_scale = {b_scale:.4f}")  
print(f"  imp = {imp:.2f}")
print(f"  Loss = {result.fun:.4f}")

# ============================================================
# 5. 运行最终模型
# ============================================================
print("\n[3] 运行模型...")

L0 = 1e7 * bi_normalized[0]
P0 = 5e6
M0 = 1e6
y0 = [L0, P0, M0*0.97, M0*0.02, M0*0.01, 200, 100]

t = np.arange(n_weeks) * 7
sol = odeint(ode_system, y0, t,
             args=(result.x, weekly_temp, bi_normalized, n_weeks))

L, P, S_m, E_m, I_m, E_h, I_h = sol.T
M_total = S_m + E_m + I_m
weekly_pred = GAMMA * I_h * 7
weekly_pred = np.maximum(weekly_pred, 0)

# ============================================================
# 6. 评估
# ============================================================
print("\n[4] 模型评估...")

corr_cases, pval = pearsonr(weekly_cases, weekly_pred)
r2_log = r2_score(np.log1p(weekly_cases), np.log1p(weekly_pred))

L_norm = L / L.mean()
corr_bi_L, _ = pearsonr(bi_normalized, L_norm)

print(f"  病例相关: r = {corr_cases:.4f}")
print(f"  R² (对数): {r2_log:.4f}")
print(f"  BI-幼虫相关: {corr_bi_L:.4f}")

# R0
R0_arr = []
for i in range(n_weeks):
    T = weekly_temp[i]
    M = M_total[i]
    b_vh = b_vh_T(T) * b_scale
    b_hv = b_hv_T(T) * b_scale
    mu_m = mu_m_T(T)
    R0 = np.sqrt(b_vh * b_hv * M / (mu_m * GAMMA * N_H + 1e-10))
    R0_arr.append(R0)
R0_arr = np.array(R0_arr)

print(f"  R0: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]")
print(f"  R0>1 周: {(R0_arr > 1).sum()}/{n_weeks}")

# ============================================================
# 7. 可视化
# ============================================================
print("\n[5] 生成图形...")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

# 1. 病例
ax = axes[0, 0]
ax.plot(weekly_cases, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax.plot(weekly_pred, 'r-', lw=1.5, label='Model', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Cases')
ax.set_title(f'Weekly Cases (r={corr_cases:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 对数
ax = axes[0, 1]
ax.semilogy(weekly_cases + 1, 'b-', lw=1.5, label='Observed')
ax.semilogy(weekly_pred + 1, 'r-', lw=1.5, label='Model')
ax.set_xlabel('Week')
ax.set_ylabel('Cases (log)')
ax.set_title(f'Log Scale (R²={r2_log:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. R0
ax = axes[0, 2]
ax.plot(R0_arr, 'g-', lw=1.5)
ax.axhline(y=1, color='red', ls='--', lw=2)
ax.fill_between(range(n_weeks), 0, R0_arr, where=R0_arr > 1, alpha=0.3, color='red')
ax.set_xlabel('Week')
ax.set_ylabel('R0')
ax.set_title('Reproduction Number')
ax.grid(True, alpha=0.3)

# 4. BI vs 幼虫
ax = axes[1, 0]
ax.plot(bi_normalized, 'g-', lw=1.5, label='BI (normalized)', alpha=0.8)
ax.plot(L_norm, 'b-', lw=1.5, label='L (model)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Normalized')
ax.set_title(f'BI vs Larvae (r={corr_bi_L:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. 水生阶段
ax = axes[1, 1]
ax.plot(L / 1e6, 'b-', lw=1, label='L (larvae)', alpha=0.7)
ax.plot(P / 1e6, 'y-', lw=1, label='P (pupae)', alpha=0.7)
ax.set_xlabel('Week')
ax.set_ylabel('Population (millions)')
ax.set_title('Aquatic Stages')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 成蚊
ax = axes[1, 2]
ax.plot(M_total / 1e6, 'g-', lw=1.5, label='Total', alpha=0.8)
ax.plot(I_m / 1e3, 'r-', lw=1.5, label='I_m (×1000)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Population')
ax.set_title('Adult Mosquitoes')
ax.legend()
ax.grid(True, alpha=0.3)

# 7. 人群
ax = axes[2, 0]
ax.plot(E_h, 'y-', lw=1.5, label='E_h', alpha=0.8)
ax.plot(I_h, 'r-', lw=1.5, label='I_h', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Population')
ax.set_title('Human E & I')
ax.legend()
ax.grid(True, alpha=0.3)

# 8. 散点图
ax = axes[2, 1]
ax.scatter(weekly_cases, weekly_pred, alpha=0.5, s=30)
max_val = max(weekly_cases.max(), weekly_pred.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Scatter')
ax.grid(True, alpha=0.3)

# 9. 温度
ax = axes[2, 2]
ax.plot(weekly_temp, 'orange', lw=1.5)
ax2 = ax.twinx()
ax2.plot(weekly_cases, 'b-', lw=1, alpha=0.5)
ax.set_xlabel('Week')
ax.set_ylabel('Temperature (°C)', color='orange')
ax2.set_ylabel('Cases', color='blue')
ax.set_title('Temperature & Cases')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/full_sei_seir_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: results/figures/full_sei_seir_model.png")

# 保存结果
pd.DataFrame({
    'week': range(n_weeks),
    'temperature': weekly_temp,
    'bi': weekly_bi,
    'bi_normalized': bi_normalized,
    'observed_cases': weekly_cases,
    'predicted_cases': weekly_pred,
    'L': L, 'P': P,
    'M_total': M_total, 'I_m': I_m,
    'E_h': E_h, 'I_h': I_h,
    'R0': R0_arr
}).to_csv('/root/wenmei/results/data/full_sei_seir_results.csv', index=False)
print("  已保存: results/data/full_sei_seir_results.csv")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("模型总结")
print("=" * 60)
print(f"""
【模型结构】
  蚊虫: L (幼虫) → P (蛹) → M (成蚊 SEI)
  人群: S_h → E_h → I_h → R_h
  
【BI 校正】
  BI 作为幼虫密度的相对指标
  公式: dL/dt 中的产卵项 × (BI/BI_mean)
  相关性: BI-幼虫 r = {corr_bi_L:.4f}

【估计参数】
  k_scale = {k_scale:.4f} (缩放)
  b_scale = {b_scale:.4f} (传播系数)
  imp = {imp:.2f} /周 (输入病例)

【性能】
  病例相关: r = {corr_cases:.4f}
  R² (对数): {r2_log:.4f}
  R0: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]

【滞后关系】
  环境 → 幼虫(L) → 蛹(P) → 成蚊(M) → 人感染(I_h)
  自然产生约 2-4 周滞后
""")
print("完成!")
