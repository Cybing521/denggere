#!/usr/bin/env python3
"""
快速版 SEI-SEIR 模型 v2
实时进度显示 + 快速优化
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
import sys
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("快速版 SEI-SEIR v2 (实时进度)")
log("=" * 60)

# ============================================================
# 1. 温度函数
# ============================================================
def phi_T(T):
    return max(0.1, 4.0 * np.exp(-((T - 28) / 7)**2))

def f_l_T(T):
    return max(0.01, 0.10 * np.exp(-((T - 27) / 9)**2))

def f_p_T(T):
    return max(0.01, 0.15 * np.exp(-((T - 27) / 9)**2))

def mu_m_T(T):
    return 0.04 + 0.002 * (T - 26)**2

def b_T(T):
    if T < 14 or T > 35:
        return 0.01
    return 0.4 * np.exp(-((T - 27) / 6)**2)

SIGMA = 0.6
EPSILON = 0.1
DELTA = 1/5
GAMMA = 1/7
N_H = 14_000_000

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
weekly_cases = np.maximum(0, cases_interp(weekly_idx))

bi_mean = weekly_bi.mean()
bi_normalized = weekly_bi / bi_mean

log(f"  周数: {n_weeks}, 总病例: {weekly_cases.sum():.0f}")
log(f"  温度: [{weekly_temp.min():.1f}, {weekly_temp.max():.1f}]°C")
log(f"  BI: [{weekly_bi.min():.2f}, {weekly_bi.max():.2f}]")

# ============================================================
# 3. ODE系统 (简化)
# ============================================================
def ode_system(y, t, params, temp_arr, bi_arr, n_weeks):
    L, P, S_m, E_m, I_m, E_h, I_h = y
    
    idx = min(int(t / 7), n_weeks - 1)
    T = temp_arr[idx]
    bi_ratio = bi_arr[idx]
    
    k_scale, b_scale, imp = params
    
    phi = phi_T(T) * k_scale
    f_l = f_l_T(T)
    f_p = f_p_T(T)
    mu_m = mu_m_T(T)
    b = b_T(T) * b_scale
    
    K_L = 1e8 * k_scale * bi_ratio
    M = max(S_m + E_m + I_m, 1)
    
    dL = phi * SIGMA * M * bi_ratio - f_l * L - 0.1 * L * (1 + L/K_L)
    dP = f_l * L - f_p * P - 0.05 * P
    
    emergence = SIGMA * f_p * P
    inf_m = b * (I_h + imp) / N_H
    
    dS_m = emergence - inf_m * S_m - mu_m * S_m
    dE_m = inf_m * S_m - EPSILON * E_m - mu_m * E_m
    dI_m = EPSILON * E_m - mu_m * I_m
    
    inf_h = b * I_m / N_H
    S_h = N_H - E_h - I_h
    
    dE_h = inf_h * S_h - DELTA * E_h
    dI_h = DELTA * E_h - GAMMA * I_h
    
    return [dL, dP, dS_m, dE_m, dI_m, dE_h, dI_h]


def run_model(params):
    k_scale, b_scale, imp = params
    
    L0 = 1e7 * bi_normalized[0] * k_scale
    P0 = L0 * 0.3
    M0 = P0 * 0.3
    I_h0 = max(weekly_cases[0], 1) / GAMMA
    E_h0 = I_h0 * 0.5
    
    y0 = [L0, P0, M0*0.95, M0*0.03, M0*0.02, E_h0, I_h0]
    t = np.arange(n_weeks) * 7
    
    try:
        sol = odeint(ode_system, y0, t,
                     args=(params, weekly_temp, bi_normalized, n_weeks),
                     rtol=1e-5, atol=1e-8)
        I_h = sol[:, 6]
        pred = GAMMA * I_h * 7
        return sol, np.maximum(pred, 0)
    except:
        return None, None


def objective(params):
    sol, pred = run_model(params)
    if pred is None:
        return 1e10
    
    k_scale, b_scale, imp = params
    
    # 主要目标：对数MSE
    mse = np.mean((np.log1p(weekly_cases) - np.log1p(pred))**2)
    
    # 相关性奖励
    if np.std(pred) > 1:
        corr, _ = pearsonr(weekly_cases, pred)
        if not np.isnan(corr) and corr > 0:
            mse -= 0.6 * corr
    
    # 峰值时间匹配奖励
    obs_peak = np.argmax(weekly_cases)
    pred_peak = np.argmax(pred)
    peak_match = 1 - abs(obs_peak - pred_peak) / n_weeks
    mse -= 0.1 * peak_match
    
    # 软约束：输入病例不超过100/周
    if imp > 100:
        mse += 0.002 * (imp - 100)
    
    return mse

# ============================================================
# 4. 网格搜索 + 局部优化
# ============================================================
log("\n[2] 网格搜索...")

# 更宽的搜索范围
k_vals = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
b_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
imp_vals = [1, 5, 10, 20, 50]

best_loss = 1e10
best_params = None
total = len(k_vals) * len(b_vals) * len(imp_vals)
count = 0

for k in k_vals:
    for b in b_vals:
        for imp in imp_vals:
            count += 1
            loss = objective([k, b, imp])
            if loss < best_loss:
                best_loss = loss
                best_params = [k, b, imp]
            if count % 25 == 0 or count == total:
                log(f"  进度: {count}/{total}, 当前最优loss={best_loss:.4f}, params=[k={best_params[0]:.1f}, b={best_params[1]:.2f}, imp={best_params[2]:.1f}]")

log(f"  网格搜索完成: loss={best_loss:.4f}, params={best_params}")

# 局部优化
log("\n[3] 局部优化...")

def callback(xk):
    loss = objective(xk)
    log(f"  优化: loss={loss:.4f}, k={xk[0]:.2f}, b={xk[1]:.3f}, imp={xk[2]:.1f}")

result = minimize(
    objective, best_params,
    method='Nelder-Mead',
    options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4},
    callback=callback
)

k_scale, b_scale, imp = result.x
log(f"\n  最终参数:")
log(f"    k_scale = {k_scale:.4f}")
log(f"    b_scale = {b_scale:.4f}")
log(f"    imp = {imp:.2f}")
log(f"    Loss = {result.fun:.4f}")

# ============================================================
# 5. 运行最终模型
# ============================================================
log("\n[4] 运行最终模型...")

sol, weekly_pred = run_model(result.x)
L, P, S_m, E_m, I_m, E_h, I_h = sol.T
M_total = S_m + E_m + I_m

# ============================================================
# 6. 评估
# ============================================================
log("\n[5] 评估...")

corr, pval = pearsonr(weekly_cases, weekly_pred)
r2_log = r2_score(np.log1p(weekly_cases), np.log1p(weekly_pred))
L_norm = L / L.mean()
corr_bi_L, _ = pearsonr(bi_normalized, L_norm)

# R0
R0_arr = []
for i in range(n_weeks):
    T = weekly_temp[i]
    M = M_total[i]
    b = b_T(T) * b_scale
    mu = mu_m_T(T)
    R0 = np.sqrt(b**2 * M / (mu * GAMMA * N_H + 1e-10))
    R0_arr.append(R0)
R0_arr = np.array(R0_arr)

log(f"  Pearson r = {corr:.4f} (p={pval:.2e})")
log(f"  R² (log) = {r2_log:.4f}")
log(f"  BI-L相关 = {corr_bi_L:.4f}")
log(f"  R0: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]")
log(f"  R0>1 周: {(R0_arr > 1).sum()}/{n_weeks}")

# ============================================================
# 7. 可视化
# ============================================================
log("\n[6] 生成图形...")

fig, axes = plt.subplots(3, 3, figsize=(16, 14))

ax = axes[0, 0]
ax.plot(weekly_cases, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax.plot(weekly_pred, 'r-', lw=1.5, label='Model', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Cases')
ax.set_title(f'Weekly Cases (r={corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.semilogy(weekly_cases + 1, 'b-', lw=1.5, label='Observed')
ax.semilogy(weekly_pred + 1, 'r-', lw=1.5, label='Model')
ax.set_xlabel('Week')
ax.set_ylabel('Cases (log)')
ax.set_title(f'Log Scale (R²={r2_log:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.plot(R0_arr, 'g-', lw=1.5)
ax.axhline(y=1, color='red', ls='--', lw=2)
ax.fill_between(range(n_weeks), 0, R0_arr, where=R0_arr > 1, alpha=0.3, color='red')
ax.set_xlabel('Week')
ax.set_ylabel('R0')
ax.set_title('Reproduction Number')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(bi_normalized, 'g-', lw=1.5, label='BI', alpha=0.8)
ax.plot(L_norm, 'b-', lw=1.5, label='L (model)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Normalized')
ax.set_title(f'BI vs Larvae (r={corr_bi_L:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(L / 1e6, 'b-', lw=1, label='L', alpha=0.7)
ax.plot(P / 1e6, 'y-', lw=1, label='P', alpha=0.7)
ax.set_xlabel('Week')
ax.set_ylabel('Population (M)')
ax.set_title('Aquatic Stages')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.plot(M_total / 1e6, 'g-', lw=1.5)
ax.set_xlabel('Week')
ax.set_ylabel('Mosquitoes (M)')
ax.set_title('Adult Mosquitoes')
ax.grid(True, alpha=0.3)

ax = axes[2, 0]
ax.plot(E_h, 'y-', lw=1.5, label='E_h', alpha=0.8)
ax.plot(I_h, 'r-', lw=1.5, label='I_h', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Population')
ax.set_title('Human E & I')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
ax.scatter(weekly_cases, weekly_pred, alpha=0.5, s=30)
max_val = max(weekly_cases.max(), weekly_pred.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Scatter')
ax.grid(True, alpha=0.3)

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
plt.savefig('/root/wenmei/results/figures/sei_seir_v2.png', dpi=150, bbox_inches='tight')
plt.close()
log("  已保存: results/figures/sei_seir_v2.png")

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
}).to_csv('/root/wenmei/results/data/sei_seir_v2.csv', index=False)
log("  已保存: results/data/sei_seir_v2.csv")

# ============================================================
# 总结
# ============================================================
log("\n" + "=" * 60)
log("模型总结")
log("=" * 60)
log(f"""
【参数】
  k_scale = {k_scale:.4f}
  b_scale = {b_scale:.4f}
  imp = {imp:.2f}

【性能】
  Pearson r = {corr:.4f}
  R² (log) = {r2_log:.4f}
  BI-L相关 = {corr_bi_L:.4f}
  R0: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]
""")
log("完成!")
