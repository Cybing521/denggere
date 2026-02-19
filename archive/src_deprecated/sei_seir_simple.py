#!/usr/bin/env python3
"""
简化版 SEI-SEIR 模型
参考: Zhang et al. (2025) PLOS NTD
"""
import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

log("=" * 60)
log("简化 SEI-SEIR 模型")
log("=" * 60)

# ============================================================
# 1. 温度依赖参数 (来自文献)
# ============================================================
def mu_m(T):
    """成蚊死亡率"""
    if T < 10 or T > 40:
        return 0.5
    return max(0.02, 0.0006 * T**2 - 0.028 * T + 0.37)

def a_rate(T):
    """叮咬率"""
    if T < 14 or T > 35:
        return 0
    return max(0, 0.0005 * T * (T - 14) * np.sqrt(35 - T))

def b_rate(T):
    """感染概率 蚊→人"""
    if T < 17 or T > 36:
        return 0
    return max(0, 0.0008 * T * (T - 17) * np.sqrt(36 - T))

# 固定参数
DELTA = 1/5     # 人潜伏期转化率
GAMMA = 1/6     # 康复率
N_H = 14_000_000

# ============================================================
# 2. 加载数据
# ============================================================
log("\n[1] 加载数据...")

case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()

log(f"  月数据: {len(df)}个月, 病例: {df['cases'].sum():,}")

# 转周数据 (简化: 每月4周)
n_weeks = len(df) * 4
weekly_temp = np.repeat(df['temperature'].values, 4)
weekly_cases = np.repeat(df['cases'].values / 4, 4)

log(f"  周数据: {n_weeks}周")

# ============================================================
# 3. 简化SEIR模型 (不含蚊虫显式动态)
# ============================================================
def seir_ode(y, t, T_arr, beta_scale, imp_rate):
    """简化SEIR: 蚊虫影响通过温度依赖的β体现"""
    S, E, I, R = y
    
    # 当前温度
    t_idx = min(int(t / 7), len(T_arr) - 1)
    T = T_arr[t_idx]
    
    # 温度依赖的有效传播率
    beta_t = beta_scale * a_rate(T) * b_rate(T) / (mu_m(T) + 1e-6)
    
    # 感染力
    lam = beta_t * I / N_H
    
    # SEIR
    dS = -lam * S
    dE = lam * S + imp_rate - DELTA * E
    dI = DELTA * E - GAMMA * I
    dR = GAMMA * I
    
    return [dS, dE, dI, dR]

# ============================================================
# 4. 参数估计
# ============================================================
log("\n[2] 参数估计...")
log("  待估计: beta_scale, imp_rate")

time_days = np.arange(n_weeks) * 7
observed = weekly_cases

def objective(params):
    beta_scale, imp_rate, I0_log = params
    I0 = 10 ** I0_log
    
    y0 = [N_H - I0*2, I0, I0, 0]
    
    try:
        sol = odeint(seir_ode, y0, time_days, args=(weekly_temp, beta_scale, imp_rate))
        I_h = sol[:, 2]
        weekly_new = GAMMA * I_h * 7
        
        # 对数MSE
        obs_log = np.log1p(observed)
        pred_log = np.log1p(weekly_new)
        mse = np.mean((obs_log - pred_log) ** 2)
        
        return mse
    except:
        return 1e10

# 简单网格搜索
log("  网格搜索...")
best_loss = 1e10
best_params = None

for beta in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    for imp in [0, 1, 5, 10, 20]:
        for I0_log in [-6, -5, -4]:
            loss = objective([beta, imp, I0_log])
            if loss < best_loss:
                best_loss = loss
                best_params = [beta, imp, I0_log]
                log(f"    beta={beta}, imp={imp}, loss={loss:.4f}")

beta_scale, imp_rate, I0_log = best_params
I0 = 10 ** I0_log

log(f"\n  最优参数:")
log(f"    beta_scale = {beta_scale}")
log(f"    imp_rate = {imp_rate}")
log(f"    I0 = {I0:.2e}")
log(f"    Loss = {best_loss:.4f}")

# ============================================================
# 5. 运行模型
# ============================================================
log("\n[3] 运行模型...")

y0 = [N_H - I0*2, I0, I0, 0]
sol = odeint(seir_ode, y0, time_days, args=(weekly_temp, beta_scale, imp_rate))

S_h, E_h, I_h, R_h = sol.T
weekly_new = GAMMA * I_h * 7

# ============================================================
# 6. 评估
# ============================================================
log("\n[4] 评估...")

corr, pval = pearsonr(observed, weekly_new)
from sklearn.metrics import r2_score
r2_log = r2_score(np.log1p(observed), np.log1p(weekly_new))

log(f"  相关系数: {corr:.4f} (p={pval:.2e})")
log(f"  R² (对数): {r2_log:.4f}")

# R0估计
R0_arr = []
for T in weekly_temp:
    beta_t = beta_scale * a_rate(T) * b_rate(T) / (mu_m(T) + 1e-6)
    R0 = beta_t / GAMMA
    R0_arr.append(R0)
R0_arr = np.array(R0_arr)

log(f"  R0 范围: [{R0_arr.min():.4f}, {R0_arr.max():.4f}]")

# ============================================================
# 7. 可视化
# ============================================================
log("\n[5] 可视化...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 病例
ax = axes[0, 0]
ax.plot(observed, 'b-', lw=1, label='Observed', alpha=0.7)
ax.plot(weekly_new, 'r-', lw=1, label='SEIR', alpha=0.7)
ax.set_title(f'Weekly Cases (r={corr:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 对数
ax = axes[0, 1]
ax.semilogy(observed + 1, 'b-', lw=1, label='Observed')
ax.semilogy(weekly_new + 1, 'r-', lw=1, label='SEIR')
ax.set_title(f'Log Scale (R²={r2_log:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. R0
ax = axes[0, 2]
ax.plot(R0_arr, 'g-', lw=1)
ax.axhline(y=1, color='red', ls='--')
ax.set_title('R0(t)')
ax.grid(True, alpha=0.3)

# 4. 温度
ax = axes[1, 0]
ax.plot(weekly_temp, 'orange', lw=1)
ax.set_title('Temperature')
ax.grid(True, alpha=0.3)

# 5. 散点
ax = axes[1, 1]
ax.scatter(observed, weekly_new, alpha=0.5, s=20)
max_val = max(observed.max(), weekly_new.max())
ax.plot([0, max_val], [0, max_val], 'k--')
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
ax.set_title('Scatter')
ax.grid(True, alpha=0.3)

# 6. I_h
ax = axes[1, 2]
ax.plot(I_h, 'r-', lw=1)
ax.set_title('Infectious (I_h)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/sei_seir_simple.png', dpi=150)
plt.close()
log("  已保存: results/figures/sei_seir_simple.png")

# 保存
pd.DataFrame({
    'week': range(n_weeks),
    'temperature': weekly_temp,
    'observed': observed,
    'predicted': weekly_new,
    'R0': R0_arr,
    'I_h': I_h
}).to_csv('/root/wenmei/results/data/sei_seir_simple.csv', index=False)
log("  已保存: results/data/sei_seir_simple.csv")

# ============================================================
# 总结
# ============================================================
log("\n" + "=" * 60)
log("模型总结")
log("=" * 60)
log(f"""
【模型】简化SEIR (温度依赖传播率)

【温度依赖参数 (文献)】
  μ_m(T): 成蚊死亡率
  a(T): 叮咬率  
  b(T): 感染概率

【固定参数】
  δ = {DELTA:.3f} /天 (潜伏期转化率)
  γ = {GAMMA:.4f} /天 (康复率)

【估计参数】
  beta_scale = {beta_scale}
  imp_rate = {imp_rate}

【性能】
  r = {corr:.4f}
  R² (对数) = {r2_log:.4f}
""")

log("\n完成!")
