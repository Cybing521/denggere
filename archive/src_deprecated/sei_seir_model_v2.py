"""
SEI-SEIR 登革热传播动力学模型 v2
参考: Zhang et al. (2025) PLOS NTD

关键特点:
1. 生物学参数使用温度依赖公式 (来自文献实验)
2. 只估计传播相关参数 (b_h, imp)
3. 周尺度建模
4. 气象驱动的蚊媒动态
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SEI-SEIR 登革热传播动力学模型 v2")
print("参考: Zhang et al. (2025) PLOS NTD")
print("时间尺度: 周")
print("=" * 70)

# ============================================================
# 1. 温度依赖的生物学参数 (Table 2公式)
# ============================================================
class TempDependentParams:
    """
    温度依赖的生物学参数
    所有公式来自文献实验数据
    """
    
    @staticmethod
    def phi(T):
        """
        产卵率 φ(T) - 公式(2)
        来自 [10]
        """
        if T < 14 or T > 35:
            return 0
        return max(0, -0.02 * T**2 + 1.1 * T - 8.5)
    
    @staticmethod
    def f_e(T):
        """
        卵孵化率 f_e(T) - 公式(3)
        来自 [11]
        """
        if T < 10 or T > 40:
            return 0
        return max(0, 0.0022 * T * (T - 10) * np.sqrt(40 - T))
    
    @staticmethod
    def f_l(T):
        """
        幼虫发育率 f_l(T) - 公式(4)
        来自 [11]
        """
        if T < 10 or T > 40:
            return 0
        return max(0, 0.00085 * T * (T - 10) * np.sqrt(40 - T))
    
    @staticmethod
    def f_p(T):
        """
        蛹发育率 f_p(T) - 公式(5)
        来自 [11]
        """
        if T < 10 or T > 40:
            return 0
        return max(0, 0.0021 * T * (T - 10) * np.sqrt(40 - T))
    
    @staticmethod
    def mu_l(T):
        """
        幼虫死亡率 μ_l(T) - 公式(6)
        来自 [11]
        """
        if T < 10 or T > 40:
            return 1.0
        return max(0.01, 0.0025 * T**2 - 0.094 * T + 0.96)
    
    @staticmethod
    def mu_p(T):
        """
        蛹死亡率 μ_p(T) - 公式(7)
        来自 [11]
        """
        if T < 10 or T > 40:
            return 1.0
        return max(0.01, 0.0003 * T**2 - 0.0126 * T + 0.14)
    
    @staticmethod
    def mu_m(T):
        """
        成蚊死亡率 μ_m(T) - 公式(8)
        来自 [10]
        """
        if T < 10 or T > 40:
            return 1.0
        return max(0.01, 0.0006 * T**2 - 0.028 * T + 0.37)
    
    @staticmethod
    def a(T):
        """
        叮咬率 a(T) - 公式(9)
        来自 [12]
        """
        if T < 14 or T > 35:
            return 0
        return max(0, 0.0005 * T * (T - 14) * np.sqrt(35 - T))
    
    @staticmethod
    def b(T):
        """
        感染概率 (蚊→人) b(T) - 公式(10)
        来自 [13]
        """
        if T < 17.05 or T > 35.83:
            return 0
        return max(0, 0.0008 * T * (T - 17.05) * np.sqrt(35.83 - T))
    
    @staticmethod
    def c(T):
        """
        感染概率 (人→蚊) c(T) - 公式(11)
        来自 [13]
        """
        if T < 12.22 or T > 37.46:
            return 0
        return max(0, 0.0007 * T * (T - 12.22) * np.sqrt(37.46 - T))


# 固定参数 (Table 2)
MU_E = 0.1        # 蚊卵死亡率 [8]
MU_EM = 0.1       # 羽化死亡率 [9]
B_V = 1.0         # 有效传播系数 (人→蚊) [14]
SIGMA = 0.5       # 雌蚊比例 [10]
DELTA = 1/5       # 人潜伏期转化率 (1/5天) [15]
GAMMA = 1/6       # 康复率 (1/6天) [15]


# ============================================================
# 2. 数据加载
# ============================================================
print("\n[1] 加载数据...")

case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
case_df['date'] = pd.to_datetime(case_df['date'])

bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
gz_bi.columns = ['year', 'month', 'bi']

# 2015-2019年数据
df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()
df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')

# 缺失BI用温度估计
for idx in df[df['bi'].isna()].index:
    temp = df.loc[idx, 'temperature']
    df.loc[idx, 'bi'] = 3 * np.exp(-((temp - 27) / 8) ** 2)

print(f"  月数据: {len(df)}个月")
print(f"  时间: {df['year'].min()}-{df['year'].max()}")
print(f"  总病例: {df['cases'].sum():,}")

# 转周数据
n_months = len(df)
n_weeks = int(n_months * 4.33)
monthly_t = np.arange(n_months) * 4.33
weekly_t = np.arange(n_weeks)

temp_interp = interp1d(monthly_t, df['temperature'].values, kind='cubic', fill_value='extrapolate')
bi_interp = interp1d(monthly_t, df['bi'].values, kind='cubic', fill_value='extrapolate')
cases_interp = interp1d(monthly_t, df['cases'].values / 4.33, kind='linear', fill_value='extrapolate')

weekly = pd.DataFrame({
    'week': weekly_t,
    'temperature': temp_interp(weekly_t),
    'bi': np.maximum(0.1, bi_interp(weekly_t)),
    'cases': np.maximum(0, cases_interp(weekly_t))
})

print(f"  周数据: {len(weekly)}周")
print(f"  温度范围: {weekly['temperature'].min():.1f} - {weekly['temperature'].max():.1f}°C")

# ============================================================
# 3. 计算温度依赖参数
# ============================================================
print("\n[2] 计算温度依赖参数...")

params = TempDependentParams()
T_values = weekly['temperature'].values

# 计算各参数的周平均值
phi_t = np.array([params.phi(T) for T in T_values])
mu_m_t = np.array([params.mu_m(T) for T in T_values])
a_t = np.array([params.a(T) for T in T_values])
b_t = np.array([params.b(T) for T in T_values])
c_t = np.array([params.c(T) for T in T_values])

print(f"  产卵率 φ 范围: {phi_t.min():.3f} - {phi_t.max():.3f}")
print(f"  成蚊死亡率 μ_m 范围: {mu_m_t.min():.3f} - {mu_m_t.max():.3f}")
print(f"  叮咬率 a 范围: {a_t.min():.3f} - {a_t.max():.3f}")
print(f"  感染概率 b 范围: {b_t.min():.3f} - {b_t.max():.3f}")
print(f"  感染概率 c 范围: {c_t.min():.3f} - {c_t.max():.3f}")


# ============================================================
# 4. 简化的SEI-SEIR模型 (周尺度)
# ============================================================
def sei_seir_weekly(t, y, T_func, b_h, imp, N_h):
    """
    SEI-SEIR 周尺度方程
    
    状态: [M, S_h, E_h, I_h, R_h]
    M: 成蚊数量 (简化处理)
    
    简化:
    - 蚊虫生命周期快，用准稳态近似
    - 成蚊数量 M 由温度驱动
    """
    M, S_h, E_h, I_h, R_h = y
    
    # 当前温度
    T = T_func(t)
    
    # 温度依赖参数
    phi_val = params.phi(T)
    mu_m_val = max(0.01, params.mu_m(T))
    a_val = params.a(T)
    b_val = params.b(T)
    c_val = params.c(T)
    
    # 蚊虫动态 (简化: 准稳态)
    # dM/dt ≈ φ - μ_m × M
    K = N_h * 3  # 环境容纳量
    dM = phi_val * M * (1 - M/K) - mu_m_val * M
    
    # 感染蚊比例 (假设与 I_h/N_h 成正比)
    i_v = c_val * a_val * I_h / N_h  # 简化估计
    
    # 人群感染力
    # λ = b_h × a × b × M × i_v / N_h
    lambda_h = b_h * a_val * b_val * M * i_v / N_h
    
    # 输入病例
    imp_rate = imp / 7  # 周转日
    
    # 人群SEIR
    dS_h = -lambda_h * S_h
    dE_h = lambda_h * S_h + imp_rate - DELTA * E_h
    dI_h = DELTA * E_h - GAMMA * I_h
    dR_h = GAMMA * I_h
    
    return [dM, dS_h, dE_h, dI_h, dR_h]


# ============================================================
# 5. 参数估计
# ============================================================
print("\n[3] 参数估计...")
print("  待估计参数:")
print("    b_h: 有效传播系数 (蚊→人)")
print("    imp: 输入病例率")

N_h = 14_000_000  # 人口
n_weeks = len(weekly)
time_days = np.arange(n_weeks) * 7  # 周转天
observed = weekly['cases'].values

# 温度插值函数
T_func = interp1d(time_days, T_values, kind='linear', fill_value='extrapolate')


def objective(x, verbose=False):
    """目标函数"""
    b_h, imp, M0_scale, I0_log = x
    I0 = 10 ** I0_log
    
    # 初始条件
    M0 = M0_scale * N_h  # 初始蚊虫数
    S0 = N_h - I0 * 2
    E0 = I0
    R0 = 0
    
    y0 = [M0, S0, E0, I0, R0]
    
    try:
        sol = solve_ivp(
            sei_seir_weekly,
            (0, time_days[-1]),
            y0,
            args=(T_func, b_h, imp, N_h),
            t_eval=time_days,
            method='RK45',
            max_step=7.0
        )
        
        if sol.status != 0:
            return 1e10
        
        I_h = sol.y[3]
        
        # 周新增 = γ × I_h × 7
        weekly_new = GAMMA * I_h * 7
        
        # 对数MSE
        obs_log = np.log1p(observed)
        pred_log = np.log1p(weekly_new)
        mse = np.mean((obs_log - pred_log) ** 2)
        
        # 相关性
        if np.std(weekly_new) > 1e-10:
            corr, _ = pearsonr(observed, weekly_new)
            if not np.isnan(corr) and corr > 0:
                mse -= 0.3 * corr
        
        return mse
        
    except Exception as e:
        return 1e10


# 搜索
bounds = [
    (0.1, 10.0),     # b_h
    (0.0, 50.0),     # imp (周输入病例)
    (0.001, 0.1),    # M0_scale (初始蚊虫/人口)
    (-7, -4),        # log10(I0)
]

print("  优化中 (请稍候)...")

# 添加回调函数显示进度
iteration_count = [0]

def callback(xk, convergence):
    iteration_count[0] += 1
    if iteration_count[0] % 20 == 0:
        print(f"    迭代 {iteration_count[0]}...")
    return False

result = differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=150,
    tol=1e-6,
    workers=1,
    callback=callback,
    polish=True
)

b_h, imp, M0_scale, I0_log = result.x
I0 = 10 ** I0_log

print(f"\n  估计结果:")
print(f"    b_h = {b_h:.4f} (有效传播系数)")
print(f"    imp = {imp:.2f} (周输入病例)")
print(f"    M0/N_h = {M0_scale:.4f} (初始蚊虫比)")
print(f"    I0 = {I0:.2e}")
print(f"    Loss = {result.fun:.4f}")


# ============================================================
# 6. 运行最终模型
# ============================================================
print("\n[4] 运行模型...")

M0 = M0_scale * N_h
S0 = N_h - I0 * 2
E0 = I0
y0 = [M0, S0, E0, I0, 0]

sol = solve_ivp(
    sei_seir_weekly,
    (0, time_days[-1]),
    y0,
    args=(T_func, b_h, imp, N_h),
    t_eval=time_days,
    method='RK45'
)

M = sol.y[0]
S_h, E_h, I_h, R_h = sol.y[1], sol.y[2], sol.y[3], sol.y[4]

weekly_new = GAMMA * I_h * 7


# ============================================================
# 7. 评估
# ============================================================
print("\n[5] 模型评估...")

corr, pval = pearsonr(observed, weekly_new)
r2_log = r2_score(np.log1p(observed), np.log1p(weekly_new))

trend_obs = np.diff(observed) > 0
trend_pred = np.diff(weekly_new) > 0
trend_acc = np.mean(trend_obs == trend_pred)

print(f"  相关系数: {corr:.4f} (p={pval:.2e})")
print(f"  R² (对数): {r2_log:.4f}")
print(f"  趋势准确率: {trend_acc:.2%}")

# R0 估计 (简化)
# R0 ≈ b_h × a² × b × c × M / (μ_m × γ × N_h)
R0_t = np.array([
    b_h * a_t[i]**2 * b_t[i] * c_t[i] * M[i] / (mu_m_t[i] * GAMMA * N_h)
    for i in range(len(M))
])

print(f"  R0 范围: [{R0_t.min():.4f}, {R0_t.max():.4f}]")
print(f"  R0 > 1 周数: {(R0_t > 1).sum()}/{len(R0_t)}")


# ============================================================
# 8. 可视化
# ============================================================
print("\n[6] 生成可视化...")

fig = plt.figure(figsize=(18, 14))
weeks = range(len(observed))

# 1. 病例对比
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(weeks, observed, 'b-', lw=1.5, label='Observed', alpha=0.8)
ax1.plot(weeks, weekly_new, 'r-', lw=1.5, label='SEI-SEIR', alpha=0.8)
ax1.set_xlabel('Week')
ax1.set_ylabel('Cases')
ax1.set_title(f'Weekly Cases (r={corr:.3f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 对数
ax2 = fig.add_subplot(3, 3, 2)
ax2.semilogy(weeks, observed + 1, 'b-', lw=1.5, label='Observed')
ax2.semilogy(weeks, weekly_new + 1, 'r-', lw=1.5, label='SEI-SEIR')
ax2.set_xlabel('Week')
ax2.set_ylabel('Cases (log)')
ax2.set_title(f'Log Scale (R²={r2_log:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. R0
ax3 = fig.add_subplot(3, 3, 3)
ax3.plot(weeks, R0_t, 'g-', lw=1.5)
ax3.axhline(y=1, color='red', ls='--', lw=2)
ax3.fill_between(weeks, 0, R0_t, where=R0_t > 1, alpha=0.3, color='red')
ax3.set_xlabel('Week')
ax3.set_ylabel('R0(t)')
ax3.set_title('Reproduction Number')
ax3.grid(True, alpha=0.3)

# 4. 温度依赖参数
ax4 = fig.add_subplot(3, 3, 4)
ax4.plot(weeks, a_t, 'b-', lw=1, label='a (bite rate)')
ax4.plot(weeks, b_t, 'r-', lw=1, label='b (M→H)')
ax4.plot(weeks, c_t, 'g-', lw=1, label='c (H→M)')
ax4.set_xlabel('Week')
ax4.set_ylabel('Rate')
ax4.set_title('Temperature-dependent Parameters')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. 蚊虫数量
ax5 = fig.add_subplot(3, 3, 5)
ax5.plot(weeks, M / 1e6, 'm-', lw=1.5)
ax5.set_xlabel('Week')
ax5.set_ylabel('Mosquito (millions)')
ax5.set_title('Mosquito Population')
ax5.grid(True, alpha=0.3)

# 6. 温度 vs 病例
ax6 = fig.add_subplot(3, 3, 6)
ax6_twin = ax6.twinx()
ax6.plot(weeks, T_values, 'orange', lw=1.5, label='Temperature')
ax6_twin.plot(weeks, observed, 'b-', lw=1, alpha=0.7, label='Cases')
ax6.set_xlabel('Week')
ax6.set_ylabel('Temperature (°C)', color='orange')
ax6_twin.set_ylabel('Cases', color='blue')
ax6.set_title('Temperature vs Cases')
ax6.grid(True, alpha=0.3)

# 7. 散点
ax7 = fig.add_subplot(3, 3, 7)
ax7.scatter(observed, weekly_new, c=weeks, cmap='viridis', s=30, alpha=0.6)
max_val = max(observed.max(), weekly_new.max())
ax7.plot([0, max_val], [0, max_val], 'k--', lw=2)
ax7.set_xlabel('Observed')
ax7.set_ylabel('Predicted')
ax7.set_title('Scatter Plot')
ax7.grid(True, alpha=0.3)

# 8. 人群动态
ax8 = fig.add_subplot(3, 3, 8)
ax8.plot(weeks, E_h, 'y-', lw=1, label='E_h')
ax8.plot(weeks, I_h, 'r-', lw=1, label='I_h')
ax8.set_xlabel('Week')
ax8.set_ylabel('Population')
ax8.set_title('Human E and I')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. 年度
ax9 = fig.add_subplot(3, 3, 9)
weekly['predicted'] = weekly_new
weekly['year'] = 2015 + (weekly['week'] // 52).astype(int)
yearly = weekly.groupby('year').agg({'cases': 'sum', 'predicted': 'sum'}).reset_index()
x = range(len(yearly))
width = 0.35
ax9.bar([i-width/2 for i in x], yearly['cases'], width, label='Observed', color='steelblue')
ax9.bar([i+width/2 for i in x], yearly['predicted'], width, label='Predicted', color='coral')
ax9.set_xticks(x)
ax9.set_xticklabels(yearly['year'].astype(int))
ax9.set_ylabel('Annual Cases')
ax9.set_title('Annual Comparison')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/wenmei/results/figures/sei_seir_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: results/figures/sei_seir_v2.png")

# 保存结果
results_df = pd.DataFrame({
    'week': weeks,
    'temperature': T_values,
    'observed': observed,
    'predicted': weekly_new,
    'M': M,
    'I_h': I_h,
    'R0': R0_t,
    'phi': phi_t,
    'mu_m': mu_m_t,
    'a': a_t,
    'b': b_t,
    'c': c_t
})
results_df.to_csv('/root/wenmei/results/data/sei_seir_v2.csv', index=False)
print("  已保存: results/data/sei_seir_v2.csv")


# ============================================================
# 9. 模型总结
# ============================================================
print("\n" + "=" * 70)
print("模型总结")
print("=" * 70)
print(f"""
【模型结构】SEI-SEIR (周尺度)

【温度依赖参数 (文献公式)】
  φ(T): 产卵率
  μ_m(T): 成蚊死亡率
  a(T): 叮咬率
  b(T): 感染概率 (蚊→人)
  c(T): 感染概率 (人→蚊)

【固定参数】
  μ_e = {MU_E} (蚊卵死亡率)
  δ = {DELTA:.3f} /天 (人潜伏期转化率, 1/5天)
  γ = {GAMMA:.4f} /天 (康复率, 1/6天)
  N_h = {N_h:,}

【估计参数】
  b_h = {b_h:.4f} (有效传播系数)
  imp = {imp:.2f} /周 (输入病例)

【模型性能】
  相关系数: {corr:.4f}
  R² (对数): {r2_log:.4f}
  趋势准确率: {trend_acc:.2%}

【下一步: 符号回归】
  可以探索 b_h 与其他因素的关系
""")

print("\n完成!")
