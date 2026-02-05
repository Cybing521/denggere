"""
SEI-SEIR 登革热传播动力学模型 (修正版)

修正要点:
1. 正确的蚊虫-人口比例 (m = N_v/N_h)
2. 使用Ross-MacDonald R0公式
3. 直接用BI作为蚊虫密度代理，不需要TCN中间步骤
4. 传播率β作为待估计参数

参考: 表4的参数设置
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 模型参数 (表4)
# ============================================================
class Params:
    """SEI-SEIR模型参数"""
    # 蚊虫参数 (文献值，固定)
    mu_v = 0.05          # 蚊虫死亡率 1/天 (寿命≈20天)
    sigma_v = 0.1        # EIP转化率 1/天 (EIP≈10天)
    b = 0.5              # 叮咬率 次/天
    
    # 传播概率 (文献值，固定)
    beta_v = 0.5         # 人→蚊传播概率
    beta_h = 0.75        # 蚊→人传播概率
    
    # 人群参数 (文献值，固定)
    sigma_h = 0.2        # 人潜伏期转化率 1/天 (潜伏期≈5天)
    gamma = 0.143        # 康复率 1/天 (感染期≈7天)
    
    # 人口
    N_h = 14_000_000     # 广州人口


# ============================================================
# 2. SEI-SEIR 微分方程 (使用蚊虫密度比 m = N_v/N_h)
# ============================================================
def sei_seir_ode(t, y, m_func, params):
    """
    SEI-SEIR 方程组 (使用蚊虫-人口比)
    
    状态变量 (比例形式):
        s_v, e_v, i_v: 蚊虫易感、暴露、感染比例
        s_h, e_h, i_h, r_h: 人群比例
    
    关键: 感染力与蚊虫密度比 m(t) 成正比
    """
    s_v, e_v, i_v, s_h, e_h, i_h, r_h = y
    
    # 蚊虫密度比 (时变)
    m_t = m_func(t)
    
    # 参数
    mu_v = params.mu_v
    sigma_v = params.sigma_v
    sigma_h = params.sigma_h
    gamma = params.gamma
    b = params.b
    beta_v = params.beta_v
    beta_h = params.beta_h
    
    # 蚊虫 SEI (比例形式)
    # Λ_v/N_v = μ_v (平衡态假设)
    ds_v = mu_v * (1 - s_v) - b * beta_v * s_v * i_h - mu_v * s_v + mu_v
    de_v = b * beta_v * s_v * i_h - sigma_v * e_v - mu_v * e_v
    di_v = sigma_v * e_v - mu_v * i_v
    
    # 简化: 假设蚊虫快速达到准平衡
    ds_v = mu_v - b * beta_v * s_v * i_h - mu_v * s_v
    de_v = b * beta_v * s_v * i_h - (sigma_v + mu_v) * e_v
    di_v = sigma_v * e_v - mu_v * i_v
    
    # 人群 SEIR
    # 感染力: λ = m(t) × b × β_h × i_v
    lambda_h = m_t * b * beta_h * i_v
    
    ds_h = -lambda_h * s_h
    de_h = lambda_h * s_h - sigma_h * e_h
    di_h = sigma_h * e_h - gamma * i_h
    dr_h = gamma * i_h
    
    return [ds_v, de_v, di_v, ds_h, de_h, di_h, dr_h]


def compute_R0_ross_macdonald(m, params):
    """
    Ross-MacDonald R0公式
    
    R0 = m × a² × b × c × e^(-μτ) / (μ × γ)
    
    简化版:
    R0 = m × b² × β_v × β_h × σ_v / (μ_v × (σ_v + μ_v) × γ)
    """
    b = params.b
    beta_v = params.beta_v
    beta_h = params.beta_h
    mu_v = params.mu_v
    sigma_v = params.sigma_v
    gamma = params.gamma
    
    # 考虑EIP存活概率
    p_survive_eip = sigma_v / (sigma_v + mu_v)
    
    R0 = m * (b**2) * beta_v * beta_h * p_survive_eip / (mu_v * gamma)
    return R0


# ============================================================
# 3. 数据准备
# ============================================================
def load_data():
    """加载2015-2019年数据"""
    print("\n[1] 加载数据...")
    
    # 病例数据
    case_df = pd.read_csv('/root/wenmei/data/guangdong_dengue_cases.csv')
    case_df['date'] = pd.to_datetime(case_df['date'])
    
    # BI数据
    bi_df = pd.read_csv('/root/wenmei/data/BI.csv', encoding='gbk')
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    gz_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    gz_bi = gz_bi[gz_bi['Site_month'] != 'Total']
    gz_bi['Site_month'] = gz_bi['Site_month'].astype(int)
    gz_bi = gz_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    gz_bi.columns = ['year', 'month', 'bi']
    
    # 2015-2019
    df = case_df[(case_df['year'] >= 2015) & (case_df['year'] <= 2019)].copy()
    df = pd.merge(df, gz_bi, on=['year', 'month'], how='left')
    
    # 缺失BI用温度估计
    for idx in df[df['bi'].isna()].index:
        temp = df.loc[idx, 'temperature']
        # 温度适宜性曲线
        df.loc[idx, 'bi'] = 3 * np.exp(-((temp - 27) / 8) ** 2)
    
    print(f"  数据范围: 2015-2019年, {len(df)}个月")
    print(f"  总病例: {df['cases'].sum():,}")
    print(f"  BI范围: {df['bi'].min():.2f} - {df['bi'].max():.2f}")
    
    return df


def prepare_weekly_data(df):
    """月数据插值到周"""
    n_months = len(df)
    n_weeks = int(n_months * 4.33)
    
    monthly_t = np.arange(n_months) * 4.33
    weekly_t = np.arange(n_weeks)
    
    # 插值
    bi_interp = interp1d(monthly_t, df['bi'].values, kind='cubic', fill_value='extrapolate')
    temp_interp = interp1d(monthly_t, df['temperature'].values, kind='cubic', fill_value='extrapolate')
    cases_interp = interp1d(monthly_t, df['cases'].values / 4.33, kind='linear', fill_value='extrapolate')
    
    weekly = pd.DataFrame({
        'week': weekly_t,
        'bi': np.maximum(0.1, bi_interp(weekly_t)),
        'temperature': temp_interp(weekly_t),
        'cases': np.maximum(0, cases_interp(weekly_t))
    })
    
    return weekly


# ============================================================
# 4. 模型拟合
# ============================================================
def fit_model(weekly_data, params):
    """
    拟合SEI-SEIR模型
    
    待估计参数:
    - m_scale: 蚊虫密度比的缩放因子 (m = m_scale × BI)
    - i_h0: 初始感染比例
    """
    print("\n[2] 拟合模型参数...")
    
    n_weeks = len(weekly_data)
    time_days = np.arange(n_weeks) * 7  # 周转天
    
    bi_values = weekly_data['bi'].values
    observed_weekly = weekly_data['cases'].values
    
    def objective(x):
        """目标函数"""
        m_scale, i_h0_log = x
        i_h0 = 10 ** i_h0_log
        
        # 蚊虫密度比 m(t) = m_scale × BI(t)
        m_values = m_scale * bi_values
        m_func = interp1d(time_days, m_values, kind='linear', fill_value='extrapolate')
        
        # 初始条件 (比例)
        i_v0 = 0.01  # 1%感染蚊虫
        e_v0 = 0.02
        s_v0 = 1 - e_v0 - i_v0
        
        e_h0 = i_h0 * 2
        s_h0 = 1 - e_h0 - i_h0
        r_h0 = 0
        
        y0 = [s_v0, e_v0, i_v0, s_h0, e_h0, i_h0, r_h0]
        
        try:
            sol = solve_ivp(
                sei_seir_ode,
                (0, time_days[-1]),
                y0,
                args=(m_func, params),
                t_eval=time_days,
                method='RK45',
                max_step=1.0
            )
            
            if sol.status != 0:
                return 1e10
            
            # 周新增 = γ × i_h × N_h × 7
            i_h = sol.y[5]
            weekly_new = params.gamma * i_h * params.N_h * 7
            
            # 对数MSE
            obs_log = np.log1p(observed_weekly)
            pred_log = np.log1p(weekly_new)
            mse = np.mean((obs_log - pred_log) ** 2)
            
            # 相关性
            if np.std(weekly_new) > 0:
                corr, _ = pearsonr(observed_weekly, weekly_new)
                if corr < 0:
                    mse += 2 * (1 - corr)
            
            return mse
            
        except:
            return 1e10
    
    # 搜索
    bounds = [
        (0.1, 10.0),    # m_scale: BI到蚊虫密度比的转换
        (-9, -5),       # log10(i_h0)
    ]
    
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=200,
        tol=1e-7,
        workers=1,
        polish=True
    )
    
    m_scale, i_h0_log = result.x
    i_h0 = 10 ** i_h0_log
    
    print(f"  估计参数:")
    print(f"    m_scale = {m_scale:.4f}")
    print(f"    i_h0 = {i_h0:.2e}")
    print(f"    Loss = {result.fun:.4f}")
    
    return m_scale, i_h0


# ============================================================
# 5. 模型运行与评估
# ============================================================
def run_model():
    """完整运行流程"""
    
    print("=" * 70)
    print("SEI-SEIR 登革热动力学模型 (修正版)")
    print("时间尺度: 周")
    print("数据: 2015-2019年")
    print("=" * 70)
    
    params = Params()
    
    # 打印参数
    print("\n固定参数 (文献值):")
    print(f"  μ_v = {params.mu_v} /天 (蚊虫死亡率)")
    print(f"  σ_v = {params.sigma_v} /天 (EIP转化率)")
    print(f"  β_v = {params.beta_v} (人→蚊传播概率)")
    print(f"  β_h = {params.beta_h} (蚊→人传播概率)")
    print(f"  b = {params.b} /天 (叮咬率)")
    print(f"  σ_h = {params.sigma_h} /天 (人潜伏期转化率)")
    print(f"  γ = {params.gamma} /天 (康复率)")
    
    # 加载数据
    df = load_data()
    weekly = prepare_weekly_data(df)
    
    print(f"\n  周数据: {len(weekly)} 周")
    print(f"  周均病例: {weekly['cases'].mean():.1f}")
    
    # 拟合
    m_scale, i_h0 = fit_model(weekly, params)
    
    # 运行最终模型
    print("\n[3] 运行模型...")
    
    n_weeks = len(weekly)
    time_days = np.arange(n_weeks) * 7
    bi_values = weekly['bi'].values
    observed = weekly['cases'].values
    
    # m(t)
    m_values = m_scale * bi_values
    m_func = interp1d(time_days, m_values, kind='linear', fill_value='extrapolate')
    
    # 初始条件
    i_v0, e_v0 = 0.01, 0.02
    s_v0 = 1 - e_v0 - i_v0
    e_h0 = i_h0 * 2
    s_h0 = 1 - e_h0 - i_h0
    y0 = [s_v0, e_v0, i_v0, s_h0, e_h0, i_h0, 0]
    
    sol = solve_ivp(
        sei_seir_ode,
        (0, time_days[-1]),
        y0,
        args=(m_func, params),
        t_eval=time_days,
        method='RK45',
        max_step=1.0
    )
    
    # 结果
    s_v, e_v, i_v = sol.y[0], sol.y[1], sol.y[2]
    s_h, e_h, i_h, r_h = sol.y[3], sol.y[4], sol.y[5], sol.y[6]
    
    weekly_new = params.gamma * i_h * params.N_h * 7
    
    # R0(t)
    R0_t = np.array([compute_R0_ross_macdonald(m, params) for m in m_values])
    
    # 评估
    print("\n[4] 模型评估...")
    
    corr, pval = pearsonr(observed, weekly_new)
    r2_log = r2_score(np.log1p(observed), np.log1p(weekly_new))
    
    obs_trend = np.diff(observed) > 0
    pred_trend = np.diff(weekly_new) > 0
    trend_acc = np.mean(obs_trend == pred_trend)
    
    print(f"  相关系数: {corr:.4f} (p={pval:.2e})")
    print(f"  R² (对数): {r2_log:.4f}")
    print(f"  趋势准确率: {trend_acc:.2%}")
    
    print(f"\n  R0(t):")
    print(f"    范围: [{R0_t.min():.3f}, {R0_t.max():.3f}]")
    print(f"    均值: {R0_t.mean():.3f}")
    print(f"    R0>1 周数: {(R0_t > 1).sum()}/{len(R0_t)}")
    
    # 可视化
    print("\n[5] 可视化...")
    
    fig = plt.figure(figsize=(18, 14))
    weeks = range(len(observed))
    
    # 1. 病例对比
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(weeks, observed, 'b-', lw=1.5, alpha=0.8, label='Observed')
    ax1.plot(weeks, weekly_new, 'r-', lw=1.5, alpha=0.8, label='SEI-SEIR')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Weekly Cases')
    ax1.set_title(f'Weekly Dengue Cases (r={corr:.3f})')
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
    
    # 3. R0(t)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(weeks, R0_t, 'g-', lw=1.5)
    ax3.axhline(y=1, color='red', ls='--', lw=2, label='R0=1')
    ax3.fill_between(weeks, 0, R0_t, where=R0_t > 1, alpha=0.3, color='red')
    ax3.set_xlabel('Week')
    ax3.set_ylabel('R0(t)')
    ax3.set_title('Basic Reproduction Number')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. m(t) 蚊虫密度比
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(weeks, m_values, 'm-', lw=1.5)
    ax4.set_xlabel('Week')
    ax4.set_ylabel('m(t) = N_v/N_h')
    ax4.set_title(f'Mosquito-Human Ratio (m_scale={m_scale:.3f})')
    ax4.grid(True, alpha=0.3)
    
    # 5. BI vs R0
    ax5 = fig.add_subplot(3, 3, 5)
    sc = ax5.scatter(bi_values, R0_t, c=weeks, cmap='viridis', s=30, alpha=0.7)
    ax5.axhline(y=1, color='red', ls='--', lw=2)
    ax5.set_xlabel('Breteau Index')
    ax5.set_ylabel('R0')
    ax5.set_title('BI vs R0')
    plt.colorbar(sc, ax=ax5, label='Week')
    ax5.grid(True, alpha=0.3)
    
    # 6. 人群动态
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.plot(weeks, e_h * 100, 'y-', lw=1.5, label='E_h (%)')
    ax6.plot(weeks, i_h * 100, 'r-', lw=1.5, label='I_h (%)')
    ax6.set_xlabel('Week')
    ax6.set_ylabel('Proportion (%)')
    ax6.set_title('Human E and I Compartments')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 蚊虫动态
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(weeks, i_v * 100, 'r-', lw=1.5, label='I_v (%)')
    ax7.set_xlabel('Week')
    ax7.set_ylabel('Infected Proportion (%)')
    ax7.set_title('Mosquito Infected Proportion')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 散点
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.scatter(observed, weekly_new, c=weeks, cmap='viridis', s=30, alpha=0.6)
    max_val = max(observed.max(), weekly_new.max())
    ax8.plot([0, max_val], [0, max_val], 'k--', lw=2)
    ax8.set_xlabel('Observed')
    ax8.set_ylabel('Predicted')
    ax8.set_title('Scatter Plot')
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
    ax9.set_xlabel('Year')
    ax9.set_ylabel('Annual Cases')
    ax9.set_title('Annual Comparison')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/wenmei/results/figures/sei_seir_corrected.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  已保存: results/figures/sei_seir_corrected.png")
    
    # 保存结果
    results_df = pd.DataFrame({
        'week': weeks,
        'observed': observed,
        'predicted': weekly_new,
        'bi': bi_values,
        'm': m_values,
        'R0': R0_t,
        'i_v': i_v,
        'i_h': i_h
    })
    results_df.to_csv('/root/wenmei/results/data/sei_seir_corrected.csv', index=False)
    print("  已保存: results/data/sei_seir_corrected.csv")
    
    # 模型总结
    print("\n" + "=" * 70)
    print("模型总结")
    print("=" * 70)
    print(f"""
【SEI-SEIR 微分方程组】

蚊虫 (SEI):
  ds_v/dt = μ_v - b·β_v·s_v·i_h - μ_v·s_v
  de_v/dt = b·β_v·s_v·i_h - (σ_v + μ_v)·e_v
  di_v/dt = σ_v·e_v - μ_v·i_v

人群 (SEIR):
  ds_h/dt = -m(t)·b·β_h·i_v·s_h
  de_h/dt = m(t)·b·β_h·i_v·s_h - σ_h·e_h
  di_h/dt = σ_h·e_h - γ·i_h
  dr_h/dt = γ·i_h

【基本再生数 R0】

R0 = m × b² × β_v × β_h × σ_v / [μ_v × (σ_v + μ_v) × γ]

其中 m(t) = m_scale × BI(t) 是蚊虫-人口密度比

【固定参数 (文献)】
  μ_v = {params.mu_v} /天    β_v = {params.beta_v}
  σ_v = {params.sigma_v} /天   β_h = {params.beta_h}
  b = {params.b} /天         σ_h = {params.sigma_h} /天
  γ = {params.gamma} /天     N_h = {params.N_h:,}

【估计参数】
  m_scale = {m_scale:.4f} (BI到蚊虫密度比的转换)
  
  物理意义: 当BI=1时, 每100人对应约 {m_scale*100:.1f} 只蚊子

【模型性能】
  相关系数: {corr:.4f}
  R² (对数): {r2_log:.4f}
  趋势准确率: {trend_acc:.2%}
  R0范围: [{R0_t.min():.3f}, {R0_t.max():.3f}]
""")
    
    return {
        'm_scale': m_scale,
        'corr': corr,
        'r2_log': r2_log,
        'R0': R0_t
    }


if __name__ == '__main__':
    results = run_model()
