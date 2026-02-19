#!/usr/bin/env python3
"""
完整版 SEI-SEIR 模型
包含人群完整状态: S_h, E_h, I_h, R_h
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
log("完整版 SEI-SEIR 模型 (含 S_h, R_h)")
log("=" * 60)

# ============================================================
# 1. 参数定义
# ============================================================
log("\n【参数说明】")
log("-" * 60)

# --- 温度依赖参数 (基于文献) ---
def phi_T(T):
    """产卵率 (Mordecai et al. 2017)
    最优温度: 28°C, 每蚊每天约4个卵
    """
    return max(0.1, 4.0 * np.exp(-((T - 28) / 7)**2))

def f_l_T(T):
    """幼虫发育率 (Yang et al. 2009)
    最优温度: 27°C, 发育周期约10天
    """
    return max(0.01, 0.10 * np.exp(-((T - 27) / 9)**2))

def f_p_T(T):
    """蛹发育率 (Yang et al. 2009)
    最优温度: 27°C, 发育周期约7天
    """
    return max(0.01, 0.15 * np.exp(-((T - 27) / 9)**2))

def mu_m_T(T):
    """成蚊死亡率 (Brady et al. 2013)
    U形曲线: 最优26°C时死亡率最低
    """
    return 0.04 + 0.002 * (T - 26)**2

def b_T(T):
    """传播概率 (Liu-Helmersson et al. 2014)
    人-蚊传播效率, 最优温度27°C
    """
    if T < 14 or T > 35:
        return 0.01
    return 0.4 * np.exp(-((T - 27) / 6)**2)

# --- 固定参数 (基于文献) ---
SIGMA = 0.6      # 卵/蛹存活率 (Yang et al.)
EPSILON = 0.1    # EIP转化率, 1/10天 (外潜伏期约10天)
DELTA = 1/5      # 人潜伏期, 1/5天 (潜伏期约5天)
GAMMA = 1/7      # 人恢复率, 1/7天 (病程约7天)
N_H = 14_000_000 # 广州人口

log("""
【温度依赖参数】(基于文献)
  φ(T)  = 4.0·exp(-((T-28)/7)²)    产卵率 (Mordecai 2017)
  f_l(T) = 0.10·exp(-((T-27)/9)²)   幼虫发育率 (Yang 2009)
  f_p(T) = 0.15·exp(-((T-27)/9)²)   蛹发育率 (Yang 2009)
  μ_m(T) = 0.04+0.002·(T-26)²       成蚊死亡率 (Brady 2013)
  b(T)  = 0.4·exp(-((T-27)/6)²)    传播概率 (Liu-Helmersson 2014)

【固定参数】
  σ = 0.6     存活率
  ε = 0.1/天  EIP转化率 (~10天外潜伏期)
  δ = 1/5/天  人潜伏期 (~5天)
  γ = 1/7/天  人恢复率 (~7天)
  N_H = 1400万 广州人口

【待优化参数】(机器学习拟合)
  k_scale: 蚊虫繁殖/承载力缩放因子
  b_scale: 传播效率缩放因子
  imp:     输入病例数 (每周)
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
weekly_cases = np.maximum(0, cases_interp(weekly_idx))

bi_mean = weekly_bi.mean()
bi_normalized = weekly_bi / bi_mean

log(f"  周数: {n_weeks}, 总病例: {weekly_cases.sum():.0f}")
log(f"  温度: [{weekly_temp.min():.1f}, {weekly_temp.max():.1f}]°C")
log(f"  BI: [{weekly_bi.min():.2f}, {weekly_bi.max():.2f}]")

# ============================================================
# 3. 完整ODE系统 (含 S_h, R_h)
# ============================================================
def ode_system(y, t, params, temp_arr, bi_arr, n_weeks):
    """
    完整 SEI-SEIR 系统
    
    状态变量 (8个):
      蚊虫: L (幼虫), P (蛹), S_m (易感), E_m (潜伏), I_m (感染)
      人群: S_h (易感), E_h (潜伏), I_h (感染)
      注: R_h = N_H - S_h - E_h - I_h (隐式计算)
    
    微分方程:
      dL/dt = φ(T)·σ·M·(BI/BI_mean) - f_l(T)·L - μ_L·L·(1+L/K)
      dP/dt = f_l(T)·L - f_p(T)·P - μ_P·P
      dS_m/dt = σ·f_p(T)·P - λ_m·S_m - μ_m(T)·S_m
      dE_m/dt = λ_m·S_m - ε·E_m - μ_m(T)·E_m
      dI_m/dt = ε·E_m - μ_m(T)·I_m
      dS_h/dt = -λ_h·S_h
      dE_h/dt = λ_h·S_h - δ·E_h
      dI_h/dt = δ·E_h - γ·I_h
    
    其中:
      λ_m = b(T)·b_scale·(I_h + imp)/N_H  (蚊感染力)
      λ_h = b(T)·b_scale·I_m/N_H          (人感染力)
      M = S_m + E_m + I_m                  (总成蚊)
    """
    L, P, S_m, E_m, I_m, S_h, E_h, I_h = y
    
    # 时间索引
    idx = min(int(t / 7), n_weeks - 1)
    T = temp_arr[idx]
    bi_ratio = bi_arr[idx]
    
    # 解包优化参数
    k_scale, b_scale, imp = params
    
    # 温度依赖率
    phi = phi_T(T) * k_scale      # 产卵率 (缩放)
    f_l = f_l_T(T)                # 幼虫发育率
    f_p = f_p_T(T)                # 蛹发育率
    mu_m = mu_m_T(T)              # 成蚊死亡率
    b = b_T(T) * b_scale          # 传播概率 (缩放)
    
    # 承载力 (BI驱动)
    K_L = 1e8 * k_scale * bi_ratio
    
    # 总成蚊数
    M = max(S_m + E_m + I_m, 1)
    
    # ========== 蚊虫动态 ==========
    # 幼虫: 产卵 - 发育 - 死亡(密度依赖)
    dL = phi * SIGMA * M * bi_ratio - f_l * L - 0.1 * L * (1 + L/K_L)
    
    # 蛹: 幼虫发育 - 羽化 - 死亡
    dP = f_l * L - f_p * P - 0.05 * P
    
    # 成蚊羽化
    emergence = SIGMA * f_p * P
    
    # 蚊感染力 (叮咬感染者)
    lambda_m = b * (I_h + imp) / N_H
    
    # 易感成蚊: 羽化 - 感染 - 死亡
    dS_m = emergence - lambda_m * S_m - mu_m * S_m
    
    # 潜伏成蚊: 感染 - 转化 - 死亡
    dE_m = lambda_m * S_m - EPSILON * E_m - mu_m * E_m
    
    # 感染成蚊: 转化 - 死亡
    dI_m = EPSILON * E_m - mu_m * I_m
    
    # ========== 人群动态 ==========
    # 人感染力 (被感染蚊叮咬)
    lambda_h = b * I_m / N_H
    
    # 易感人群: 被感染
    dS_h = -lambda_h * S_h
    
    # 潜伏人群: 感染 - 发病
    dE_h = lambda_h * S_h - DELTA * E_h
    
    # 感染人群: 发病 - 恢复
    dI_h = DELTA * E_h - GAMMA * I_h
    
    # 注: R_h = N_H - S_h - E_h - I_h (自动守恒)
    
    return [dL, dP, dS_m, dE_m, dI_m, dS_h, dE_h, dI_h]


def run_model(params):
    """运行模型并返回结果"""
    k_scale, b_scale, imp = params
    
    # 初始条件
    L0 = 1e7 * bi_normalized[0] * k_scale
    P0 = L0 * 0.3
    M0 = P0 * 0.3
    
    # 人群初始状态
    I_h0 = max(weekly_cases[0], 1) / GAMMA
    E_h0 = I_h0 * 0.5
    S_h0 = N_H - E_h0 - I_h0  # 几乎全部易感
    
    y0 = [L0, P0, M0*0.95, M0*0.03, M0*0.02, S_h0, E_h0, I_h0]
    t = np.arange(n_weeks) * 7
    
    try:
        sol = odeint(ode_system, y0, t,
                     args=(params, weekly_temp, bi_normalized, n_weeks),
                     rtol=1e-6, atol=1e-9)
        I_h = sol[:, 7]
        pred = GAMMA * I_h * 7  # 周发病数
        return sol, np.maximum(pred, 0)
    except:
        return None, None


def objective(params):
    """优化目标函数"""
    sol, pred = run_model(params)
    if pred is None:
        return 1e10
    
    k_scale, b_scale, imp = params
    
    mse = np.mean((np.log1p(weekly_cases) - np.log1p(pred))**2)
    
    if np.std(pred) > 1:
        corr, _ = pearsonr(weekly_cases, pred)
        if not np.isnan(corr) and corr > 0:
            mse -= 0.6 * corr
    
    # 峰值匹配
    obs_peak = np.argmax(weekly_cases)
    pred_peak = np.argmax(pred)
    peak_match = 1 - abs(obs_peak - pred_peak) / n_weeks
    mse -= 0.1 * peak_match
    
    # 软约束
    if imp > 100:
        mse += 0.002 * (imp - 100)
    
    return mse

# ============================================================
# 4. 优化
# ============================================================
log("\n[2] 网格搜索...")

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
            if count % 45 == 0 or count == total:
                log(f"  进度: {count}/{total}, 最优loss={best_loss:.4f}")

log(f"  网格搜索完成: params=[k={best_params[0]:.1f}, b={best_params[1]:.2f}, imp={best_params[2]:.1f}]")

log("\n[3] 局部优化...")

def callback(xk):
    loss = objective(xk)
    if np.random.random() < 0.1:  # 10%概率打印
        log(f"  优化: loss={loss:.4f}, k={xk[0]:.2f}, b={xk[1]:.3f}, imp={xk[2]:.1f}")

result = minimize(
    objective, best_params,
    method='Nelder-Mead',
    options={'maxiter': 150, 'xatol': 1e-4, 'fatol': 1e-4},
    callback=callback
)

k_scale, b_scale, imp = result.x
log(f"\n  【优化参数】(机器学习拟合)")
log(f"    k_scale = {k_scale:.4f}")
log(f"    b_scale = {b_scale:.4f}")
log(f"    imp = {imp:.2f} /周")
log(f"    Loss = {result.fun:.4f}")

# ============================================================
# 5. 运行最终模型
# ============================================================
log("\n[4] 运行最终模型...")

sol, weekly_pred = run_model(result.x)
L, P, S_m, E_m, I_m, S_h, E_h, I_h = sol.T
M_total = S_m + E_m + I_m
R_h = N_H - S_h - E_h - I_h  # 恢复人群

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

# 人群状态统计
log(f"\n  【人群状态】")
log(f"    S_h (易感): {S_h[-1]/1e6:.2f}M → {S_h[-1]/N_H*100:.2f}%")
log(f"    累计感染: {(N_H - S_h[-1])/1e3:.1f}K")
log(f"    R_h (恢复): {R_h[-1]/1e3:.1f}K")

# ============================================================
# 7. 可视化
# ============================================================
log("\n[6] 生成图形...")

fig, axes = plt.subplots(4, 3, figsize=(18, 20))

# 1. 病例拟合
ax = axes[0, 0]
ax.plot(weekly_cases, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax.plot(weekly_pred, 'r-', lw=1.5, label='Model', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Cases')
ax.set_title(f'Weekly Cases (r={corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 对数尺度
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
ax.plot(bi_normalized, 'g-', lw=1.5, label='BI', alpha=0.8)
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
ax.set_ylabel('Population (M)')
ax.set_title('Aquatic Stages')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. 成蚊
ax = axes[1, 2]
ax.plot(M_total / 1e6, 'g-', lw=1.5, label='Total')
ax.plot(I_m / 1e3, 'r-', lw=1.5, label='I_m (×1000)')
ax.set_xlabel('Week')
ax.set_ylabel('Population')
ax.set_title('Adult Mosquitoes')
ax.legend()
ax.grid(True, alpha=0.3)

# 7. 人群 S, E, I, R
ax = axes[2, 0]
ax.plot((N_H - S_h) / 1e3, 'purple', lw=1.5, label='Cumulative Infected')
ax.plot(R_h / 1e3, 'green', lw=1.5, label='R_h (Recovered)')
ax.set_xlabel('Week')
ax.set_ylabel('Population (K)')
ax.set_title('Cumulative Infection & Recovery')
ax.legend()
ax.grid(True, alpha=0.3)

# 8. E_h, I_h
ax = axes[2, 1]
ax.plot(E_h, 'y-', lw=1.5, label='E_h (Exposed)', alpha=0.8)
ax.plot(I_h, 'r-', lw=1.5, label='I_h (Infected)', alpha=0.8)
ax.set_xlabel('Week')
ax.set_ylabel('Population')
ax.set_title('Human E & I')
ax.legend()
ax.grid(True, alpha=0.3)

# 9. 散点图
ax = axes[2, 2]
ax.scatter(weekly_cases, weekly_pred, alpha=0.5, s=30, c=range(n_weeks), cmap='viridis')
max_val = max(weekly_cases.max(), weekly_pred.max())
ax.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Scatter')
ax.grid(True, alpha=0.3)

# 10. 温度
ax = axes[3, 0]
ax.plot(weekly_temp, 'orange', lw=1.5)
ax2 = ax.twinx()
ax2.plot(weekly_cases, 'b-', lw=1, alpha=0.5)
ax.set_xlabel('Week')
ax.set_ylabel('Temperature (°C)', color='orange')
ax2.set_ylabel('Cases', color='blue')
ax.set_title('Temperature & Cases')
ax.grid(True, alpha=0.3)

# 11. 蚊虫 SEI
ax = axes[3, 1]
ax.plot(S_m / M_total, 'b-', lw=1, label='S_m/M', alpha=0.7)
ax.plot(E_m / M_total, 'y-', lw=1, label='E_m/M', alpha=0.7)
ax.plot(I_m / M_total, 'r-', lw=1, label='I_m/M', alpha=0.7)
ax.set_xlabel('Week')
ax.set_ylabel('Proportion')
ax.set_title('Mosquito SEI Proportions')
ax.legend()
ax.grid(True, alpha=0.3)

# 12. 残差
ax = axes[3, 2]
residuals = weekly_cases - weekly_pred
ax.bar(range(n_weeks), residuals, alpha=0.6, width=1)
ax.axhline(y=0, color='black', lw=1)
ax.set_xlabel('Week')
ax.set_ylabel('Residual')
ax.set_title('Residuals (Obs - Pred)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/sei_seir_complete.png', dpi=150, bbox_inches='tight')
plt.close()
log("  已保存: results/figures/sei_seir_complete.png")

# 保存结果
pd.DataFrame({
    'week': range(n_weeks),
    'temperature': weekly_temp,
    'bi': weekly_bi,
    'bi_normalized': bi_normalized,
    'observed_cases': weekly_cases,
    'predicted_cases': weekly_pred,
    'L': L, 'P': P,
    'S_m': S_m, 'E_m': E_m, 'I_m': I_m,
    'M_total': M_total,
    'S_h': S_h, 'E_h': E_h, 'I_h': I_h, 'R_h': R_h,
    'R0': R0_arr
}).to_csv('/root/wenmei/results/data/sei_seir_complete.csv', index=False)
log("  已保存: results/data/sei_seir_complete.csv")

# ============================================================
# 8. 总结
# ============================================================
log("\n" + "=" * 60)
log("完整模型总结")
log("=" * 60)
log(f"""
【模型结构】
  蚊虫: L → P → S_m → E_m → I_m (水生期 + 成蚊SEI)
  人群: S_h → E_h → I_h → R_h (完整SEIR)

【微分方程】
  dL/dt = φ(T)·σ·M·BI_ratio - f_l(T)·L - μ_L·L·(1+L/K)
  dP/dt = f_l(T)·L - f_p(T)·P - μ_P·P
  dS_m/dt = σ·f_p(T)·P - λ_m·S_m - μ_m(T)·S_m
  dE_m/dt = λ_m·S_m - ε·E_m - μ_m(T)·E_m
  dI_m/dt = ε·E_m - μ_m(T)·I_m
  dS_h/dt = -λ_h·S_h
  dE_h/dt = λ_h·S_h - δ·E_h
  dI_h/dt = δ·E_h - γ·I_h

  其中: λ_m = b(T)·(I_h+imp)/N_H, λ_h = b(T)·I_m/N_H

【参数来源】
  文献参数: φ, f_l, f_p, μ_m, b (温度依赖)
            σ=0.6, ε=0.1, δ=1/5, γ=1/7 (固定)
  
  优化参数: k_scale={k_scale:.4f}, b_scale={b_scale:.4f}, imp={imp:.2f}

【模型性能】
  Pearson r = {corr:.4f}
  R² (log) = {r2_log:.4f}
  BI-L相关 = {corr_bi_L:.4f}
  R0: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]

【人群状态】
  累计感染: {(N_H - S_h[-1])/1e3:.1f}K ({(N_H - S_h[-1])/N_H*100:.3f}%)
  恢复人群: {R_h[-1]/1e3:.1f}K
""")
log("完成!")
