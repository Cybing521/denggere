"""
SEI-SEIR 登革热传播动力学模型 (最终版)

关键改进:
1. 生物学参数固定 (文献值): μ_v, σ_v, σ_h, γ
2. 有效传播率β_eff待估计 (综合了β_v, β_h, b等)
3. 蚊虫密度m(t)由BI驱动

这样的好处:
- β_eff可以用符号回归找到与环境因素的关系
- 与PNAS论文的方法一致 (β'作为待估计参数)
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

# ============================================================
# 固定参数 (文献值)
# ============================================================
MU_V = 0.05           # 蚊虫死亡率 (1/天), 寿命≈20天
SIGMA_V = 0.1         # 蚊虫EIP转化率 (1/天), EIP≈10天
SIGMA_H = 0.2         # 人潜伏期转化率 (1/天), 潜伏期≈5天
GAMMA = 0.143         # 康复率 (1/天), 感染期≈7天
N_H = 14_000_000      # 人口


# ============================================================
# 简化的SEI-SEIR模型
# ============================================================
def sei_seir_simplified(t, y, beta_eff, m_func):
    """
    简化的SEI-SEIR方程
    
    关键简化: 
    - 蚊虫快速达到准稳态
    - 有效传播率β_eff是综合参数
    
    状态: [s_v, e_v, i_v, s_h, e_h, i_h, r_h] (比例)
    
    感染力:
    - 蚊虫被感染: β_eff × i_h
    - 人被感染: β_eff × m(t) × i_v
    """
    s_v, e_v, i_v, s_h, e_h, i_h, r_h = y
    
    m_t = m_func(t)  # 蚊虫密度比
    
    # 蚊虫 SEI
    ds_v = MU_V * (1 - s_v) - beta_eff * s_v * i_h
    de_v = beta_eff * s_v * i_h - (SIGMA_V + MU_V) * e_v
    di_v = SIGMA_V * e_v - MU_V * i_v
    
    # 人群 SEIR
    lambda_h = beta_eff * m_t * i_v  # 感染力
    
    ds_h = -lambda_h * s_h
    de_h = lambda_h * s_h - SIGMA_H * e_h
    di_h = SIGMA_H * e_h - GAMMA * i_h
    dr_h = GAMMA * i_h
    
    return [ds_v, de_v, di_v, ds_h, de_h, di_h, dr_h]


def compute_R0(beta_eff, m):
    """
    计算R0
    R0 = β_eff² × m × σ_v / [μ_v × (σ_v + μ_v) × γ]
    """
    p_survive = SIGMA_V / (SIGMA_V + MU_V)
    R0 = (beta_eff ** 2) * m * p_survive / (MU_V * GAMMA)
    return R0


# ============================================================
# 数据加载
# ============================================================
def load_data():
    """加载2015-2019数据"""
    case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
    case_df['date'] = pd.to_datetime(case_df['date'])
    
    bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
    gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
    gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    gz_bi.columns = ['year', 'month', 'bi']
    
    df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()
    df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')
    
    # 缺失BI用温度估计
    for idx in df[df['bi'].isna()].index:
        temp = df.loc[idx, 'temperature']
        df.loc[idx, 'bi'] = 3 * np.exp(-((temp - 27) / 8) ** 2)
    
    return df


def to_weekly(df):
    """月数据转周数据"""
    n_months = len(df)
    n_weeks = int(n_months * 4.33)
    
    monthly_t = np.arange(n_months) * 4.33
    weekly_t = np.arange(n_weeks)
    
    bi_interp = interp1d(monthly_t, df['bi'].values, kind='cubic', fill_value='extrapolate')
    temp_interp = interp1d(monthly_t, df['temperature'].values, kind='cubic', fill_value='extrapolate')
    cases_interp = interp1d(monthly_t, df['cases'].values / 4.33, kind='linear', fill_value='extrapolate')
    
    return pd.DataFrame({
        'week': weekly_t,
        'bi': np.maximum(0.1, bi_interp(weekly_t)),
        'temperature': temp_interp(weekly_t),
        'cases': np.maximum(0, cases_interp(weekly_t))
    })


# ============================================================
# 模型拟合
# ============================================================
def fit_model(weekly):
    """
    拟合模型
    
    待估计参数:
    - beta_eff: 有效传播率
    - m_scale: BI到蚊虫密度比的转换因子
    - i_h0: 初始感染比例
    """
    n_weeks = len(weekly)
    time_days = np.arange(n_weeks) * 7
    bi_values = weekly['bi'].values
    observed = weekly['cases'].values
    
    def objective(x):
        beta_eff, m_scale, i_h0_log = x
        i_h0 = 10 ** i_h0_log
        
        # m(t)
        m_values = m_scale * bi_values
        m_func = interp1d(time_days, m_values, kind='linear', fill_value='extrapolate')
        
        # 初始条件
        y0 = [
            0.97, 0.02, 0.01,      # s_v, e_v, i_v
            1 - 2*i_h0, i_h0, i_h0, 0  # s_h, e_h, i_h, r_h
        ]
        
        try:
            sol = solve_ivp(
                sei_seir_simplified,
                (0, time_days[-1]),
                y0,
                args=(beta_eff, m_func),
                t_eval=time_days,
                method='RK45',
                max_step=1.0
            )
            
            if sol.status != 0:
                return 1e10
            
            i_h = sol.y[5]
            weekly_new = GAMMA * i_h * N_H * 7
            
            # 对数MSE
            obs_log = np.log1p(observed)
            pred_log = np.log1p(weekly_new)
            mse = np.mean((obs_log - pred_log) ** 2)
            
            # 相关性奖励
            if np.std(weekly_new) > 1e-10:
                corr, _ = pearsonr(observed, weekly_new)
                if not np.isnan(corr):
                    mse -= 0.5 * corr  # 相关性越高，损失越低
            
            return mse
            
        except:
            return 1e10
    
    # 搜索
    bounds = [
        (0.001, 0.5),   # beta_eff (有效传播率，应该比文献值低)
        (0.01, 2.0),    # m_scale
        (-9, -5),       # log10(i_h0)
    ]
    
    print("  优化中...")
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=300,
        tol=1e-8,
        workers=1,
        polish=True
    )
    
    return result.x, result.fun


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("SEI-SEIR 登革热动力学模型")
    print("=" * 70)
    
    print("\n【固定参数 (文献)】")
    print(f"  μ_v = {MU_V} /天 (蚊虫死亡率, 寿命≈{1/MU_V:.0f}天)")
    print(f"  σ_v = {SIGMA_V} /天 (EIP转化率, EIP≈{1/SIGMA_V:.0f}天)")
    print(f"  σ_h = {SIGMA_H} /天 (人潜伏期转化率, 潜伏期≈{1/SIGMA_H:.0f}天)")
    print(f"  γ = {GAMMA} /天 (康复率, 感染期≈{1/GAMMA:.0f}天)")
    print(f"  N_h = {N_H:,}")
    
    print("\n【待估计参数】")
    print("  β_eff: 有效传播率 (综合了叮咬率、传播概率等)")
    print("  m_scale: BI到蚊虫密度比的转换")
    
    # 数据
    print("\n[1] 加载数据...")
    df = load_data()
    weekly = to_weekly(df)
    print(f"  月数据: {len(df)}个月, 周数据: {len(weekly)}周")
    print(f"  总病例: {df['cases'].sum():,.0f}")
    
    # 拟合
    print("\n[2] 参数估计...")
    params, loss = fit_model(weekly)
    beta_eff, m_scale, i_h0_log = params
    i_h0 = 10 ** i_h0_log
    
    print(f"\n  估计结果:")
    print(f"    β_eff = {beta_eff:.4f}")
    print(f"    m_scale = {m_scale:.4f}")
    print(f"    i_h0 = {i_h0:.2e}")
    print(f"    Loss = {loss:.4f}")
    
    # 运行
    print("\n[3] 模型运行...")
    n_weeks = len(weekly)
    time_days = np.arange(n_weeks) * 7
    bi_values = weekly['bi'].values
    observed = weekly['cases'].values
    
    m_values = m_scale * bi_values
    m_func = interp1d(time_days, m_values, kind='linear', fill_value='extrapolate')
    
    y0 = [0.97, 0.02, 0.01, 1 - 2*i_h0, i_h0, i_h0, 0]
    
    sol = solve_ivp(
        sei_seir_simplified,
        (0, time_days[-1]),
        y0,
        args=(beta_eff, m_func),
        t_eval=time_days,
        method='RK45'
    )
    
    s_v, e_v, i_v = sol.y[0], sol.y[1], sol.y[2]
    s_h, e_h, i_h, r_h = sol.y[3], sol.y[4], sol.y[5], sol.y[6]
    
    weekly_new = GAMMA * i_h * N_H * 7
    
    # R0
    R0_t = np.array([compute_R0(beta_eff, m) for m in m_values])
    
    # 评估
    print("\n[4] 评估...")
    corr, pval = pearsonr(observed, weekly_new)
    r2_log = r2_score(np.log1p(observed), np.log1p(weekly_new))
    
    trend_obs = np.diff(observed) > 0
    trend_pred = np.diff(weekly_new) > 0
    trend_acc = np.mean(trend_obs == trend_pred)
    
    print(f"  相关系数: {corr:.4f} (p={pval:.2e})")
    print(f"  R² (对数): {r2_log:.4f}")
    print(f"  趋势准确率: {trend_acc:.2%}")
    print(f"  R0 范围: [{R0_t.min():.3f}, {R0_t.max():.3f}]")
    print(f"  R0 > 1 周数: {(R0_t > 1).sum()}/{len(R0_t)}")
    
    # 可视化
    print("\n[5] 可视化...")
    
    fig = plt.figure(figsize=(18, 12))
    weeks = range(len(observed))
    
    # 1. 病例
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(weeks, observed, 'b-', lw=1.5, label='Observed', alpha=0.8)
    ax1.plot(weeks, weekly_new, 'r-', lw=1.5, label='SEI-SEIR', alpha=0.8)
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Weekly Cases')
    ax1.set_title(f'Weekly Cases (r={corr:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 对数
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.semilogy(weeks, observed + 1, 'b-', lw=1.5, label='Observed')
    ax2.semilogy(weeks, weekly_new + 1, 'r-', lw=1.5, label='SEI-SEIR')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Cases (log)')
    ax2.set_title(f'Log Scale (R²={r2_log:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. R0
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(weeks, R0_t, 'g-', lw=1.5)
    ax3.axhline(y=1, color='red', ls='--', lw=2, label='R0=1')
    ax3.fill_between(weeks, 0, R0_t, where=R0_t > 1, alpha=0.3, color='red')
    ax3.set_xlabel('Week')
    ax3.set_ylabel('R0(t)')
    ax3.set_title('Basic Reproduction Number')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. BI vs Cases
    ax4 = fig.add_subplot(2, 3, 4)
    ax4_twin = ax4.twinx()
    ln1 = ax4.plot(weeks, bi_values, 'g-', lw=1.5, label='BI')
    ln2 = ax4_twin.plot(weeks, observed, 'b-', lw=1.5, label='Cases', alpha=0.7)
    ax4.set_xlabel('Week')
    ax4.set_ylabel('Breteau Index', color='green')
    ax4_twin.set_ylabel('Cases', color='blue')
    ax4.set_title('BI vs Cases')
    lns = ln1 + ln2
    ax4.legend(lns, [l.get_label() for l in lns])
    ax4.grid(True, alpha=0.3)
    
    # 5. 散点
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(observed, weekly_new, c=weeks, cmap='viridis', s=30, alpha=0.6)
    max_val = max(observed.max(), weekly_new.max())
    ax5.plot([0, max_val], [0, max_val], 'k--', lw=2)
    ax5.set_xlabel('Observed')
    ax5.set_ylabel('Predicted')
    ax5.set_title('Scatter')
    ax5.grid(True, alpha=0.3)
    
    # 6. 年度
    ax6 = fig.add_subplot(2, 3, 6)
    weekly['predicted'] = weekly_new
    weekly['year'] = 2015 + (weekly['week'] // 52).astype(int)
    yearly = weekly.groupby('year').agg({'cases': 'sum', 'predicted': 'sum'}).reset_index()
    x = range(len(yearly))
    width = 0.35
    ax6.bar([i-width/2 for i in x], yearly['cases'], width, label='Observed', color='steelblue')
    ax6.bar([i+width/2 for i in x], yearly['predicted'], width, label='Predicted', color='coral')
    ax6.set_xticks(x)
    ax6.set_xticklabels(yearly['year'].astype(int))
    ax6.set_ylabel('Annual Cases')
    ax6.set_title('Annual Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/wenmei/results/figures/sei_seir_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: results/figures/sei_seir_final.png")
    
    # 保存
    results_df = pd.DataFrame({
        'week': weeks,
        'observed': observed,
        'predicted': weekly_new,
        'bi': bi_values,
        'm': m_values,
        'R0': R0_t,
        'temperature': weekly['temperature'].values,
        'i_v': i_v,
        'i_h': i_h
    })
    results_df.to_csv('/root/wenmei/results/data/sei_seir_final.csv', index=False)
    print("  已保存: results/data/sei_seir_final.csv")
    
    # 为符号回归准备数据
    symbolic_df = pd.DataFrame({
        'week': weeks,
        'temperature': weekly['temperature'].values,
        'bi': bi_values,
        'm': m_values,
        'beta_eff': beta_eff,  # 常数，但符号回归可以探索其与环境的关系
        'R0': R0_t
    })
    symbolic_df.to_csv('/root/wenmei/results/data/for_symbolic_regression.csv', index=False)
    print("  已保存: results/data/for_symbolic_regression.csv")
    
    # 总结
    print("\n" + "=" * 70)
    print("模型总结")
    print("=" * 70)
    print(f"""
【微分方程组】

蚊虫 (SEI):
  ds_v/dt = μ_v(1-s_v) - β_eff·s_v·i_h
  de_v/dt = β_eff·s_v·i_h - (σ_v+μ_v)·e_v
  di_v/dt = σ_v·e_v - μ_v·i_v

人群 (SEIR):
  ds_h/dt = -β_eff·m(t)·i_v·s_h
  de_h/dt = β_eff·m(t)·i_v·s_h - σ_h·e_h
  di_h/dt = σ_h·e_h - γ·i_h
  dr_h/dt = γ·i_h

【基本再生数】
  R0 = β_eff² × m × σ_v / [μ_v × (σ_v + μ_v) × γ]

【固定参数 (文献)】
  μ_v = {MU_V} /天     σ_v = {SIGMA_V} /天
  σ_h = {SIGMA_H} /天   γ = {GAMMA} /天

【估计参数】
  β_eff = {beta_eff:.4f}  (有效传播率)
  m_scale = {m_scale:.4f}  (BI→蚊虫密度比)

【性能】
  相关系数: {corr:.4f}
  R² (对数): {r2_log:.4f}
  趋势准确率: {trend_acc:.2%}

【下一步: 符号回归】
  目标: 找到 β_eff 或 m 与环境因素(T, H, P)的解析关系
  输入: for_symbolic_regression.csv
""")
    
    return {
        'beta_eff': beta_eff,
        'm_scale': m_scale,
        'corr': corr,
        'r2_log': r2_log,
        'R0': R0_t
    }


if __name__ == '__main__':
    results = main()
