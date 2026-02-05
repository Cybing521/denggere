"""
使用广东省实际登革热病例数据验证SEI-SEIR模型的人群动态预测
数据来源: https://github.com/xyyu001/CCM14
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入模型
from sei_seir_model import SEISEIRModel, SEISEIRParameters
from tcn_model import TCN


def load_case_data(case_path):
    """加载广东省病例数据"""
    df = pd.read_csv(case_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_bi_data(bi_path):
    """加载广州BI数据"""
    bi_df = pd.read_csv(bi_path, encoding='gbk')
    
    # 筛选广东广州数据
    gd = bi_df[bi_df['Site_L1'] == 'Guangdong']
    guangzhou_bi = gd[gd['Site_L2'] == 'Guangzhou'].copy()
    
    # 过滤掉Total行
    guangzhou_bi = guangzhou_bi[guangzhou_bi['Site_month'] != 'Total']
    guangzhou_bi['Site_month'] = guangzhou_bi['Site_month'].astype(int)
    
    # 按年月聚合 (Den_admin是BI值)
    guangzhou_bi = guangzhou_bi.groupby(['Site_year', 'Site_month'])['Den_admin'].mean().reset_index()
    guangzhou_bi.columns = ['year', 'month', 'bi']
    
    return guangzhou_bi


def merge_data(case_df, bi_df):
    """合并病例数据和BI数据"""
    # 添加年月列到病例数据
    case_df['year'] = case_df['date'].dt.year
    case_df['month'] = case_df['date'].dt.month
    
    # 合并
    merged = pd.merge(case_df, bi_df, on=['year', 'month'], how='left')
    
    return merged


class ImprovedSEISEIRSimulator:
    """改进的SEI-SEIR模拟器，包含病例预测"""
    
    def __init__(self, N_h=11000000, N_v_ratio=2.0):
        """
        初始化参数
        N_h: 人口数量 (广东省约1.1亿)
        N_v_ratio: 蚊人比
        """
        self.N_h = N_h
        self.N_v_ratio = N_v_ratio
        
        # 蚊虫参数
        self.mu_v = 1/14  # 蚊虫死亡率 (寿命14天)
        self.sigma_v = 1/10  # 蚊虫潜伏期转化率 (10天)
        self.beta_vh = 0.5  # 蚊→人传播概率
        self.beta_hv = 0.5  # 人→蚊传播概率
        self.b = 0.5  # 叮咬率
        
        # 人群参数
        self.sigma_h = 1/5.5  # 人潜伏期转化率 (5.5天)
        self.gamma_h = 1/7  # 人恢复率 (7天)
        self.mu_h = 1/(70*365)  # 人死亡率
        
    def compute_lambda_v(self, temperature, humidity, precipitation):
        """计算蚊虫出生率 (基于气象因素)"""
        # 温度适宜性 (最适温度25-28°C)
        T_opt = 26.5
        T_width = 8
        temp_factor = np.exp(-((temperature - T_opt) / T_width) ** 2)
        
        # 湿度适宜性 (最适湿度70-80%)
        H_opt = 75
        H_width = 20
        humid_factor = np.exp(-((humidity - H_opt) / H_width) ** 2)
        
        # 降水影响 (适度降水有利)
        precip_factor = 1 / (1 + np.exp(-0.5 * (precipitation - 3)))
        
        # 基础出生率
        base_lambda = 500
        
        return base_lambda * temp_factor * humid_factor * (0.5 + 0.5 * precip_factor)
    
    def simulate(self, months, weather_data, lambda_v_series, I_h_init=10):
        """
        模拟SEI-SEIR动态
        
        Parameters:
        -----------
        months: 月份数
        weather_data: 气象数据 DataFrame
        lambda_v_series: TCN预测的蚊虫出生率序列
        I_h_init: 初始感染人数
        
        Returns:
        --------
        results: 包含各仓室状态的字典
        """
        # 初始条件
        N_v = self.N_h * self.N_v_ratio
        
        S_v = N_v * 0.9
        E_v = N_v * 0.05
        I_v = N_v * 0.05
        
        S_h = self.N_h - I_h_init
        E_h = 0
        I_h = I_h_init
        R_h = 0
        
        results = {
            'S_v': [], 'E_v': [], 'I_v': [],
            'S_h': [], 'E_h': [], 'I_h': [], 'R_h': [],
            'new_cases': [], 'R0': []
        }
        
        dt = 1  # 1天步长
        days_per_month = 30
        
        for m in range(months):
            lambda_v = lambda_v_series[m] if m < len(lambda_v_series) else lambda_v_series[-1]
            
            monthly_new_cases = 0
            
            for d in range(days_per_month):
                # 计算力
                N_v_current = S_v + E_v + I_v
                
                # 蚊虫动态
                force_vh = self.b * self.beta_hv * I_h / self.N_h
                
                dS_v = lambda_v - force_vh * S_v - self.mu_v * S_v
                dE_v = force_vh * S_v - self.sigma_v * E_v - self.mu_v * E_v
                dI_v = self.sigma_v * E_v - self.mu_v * I_v
                
                # 人群动态
                force_hv = self.b * self.beta_vh * I_v / self.N_h
                
                new_exposed = force_hv * S_h
                new_infected = self.sigma_h * E_h
                
                dS_h = -force_hv * S_h
                dE_h = force_hv * S_h - self.sigma_h * E_h
                dI_h = self.sigma_h * E_h - self.gamma_h * I_h
                dR_h = self.gamma_h * I_h
                
                # 更新状态
                S_v = max(0, S_v + dS_v * dt)
                E_v = max(0, E_v + dE_v * dt)
                I_v = max(0, I_v + dI_v * dt)
                
                S_h = max(0, S_h + dS_h * dt)
                E_h = max(0, E_h + dE_h * dt)
                I_h = max(0, I_h + dI_h * dt)
                R_h = max(0, R_h + dR_h * dt)
                
                monthly_new_cases += new_infected
            
            # 计算R0
            N_v_current = S_v + E_v + I_v
            m_ratio = N_v_current / self.N_h
            R0 = (self.b ** 2 * self.beta_vh * self.beta_hv * m_ratio * self.sigma_v) / \
                 ((self.sigma_v + self.mu_v) * self.mu_v * self.gamma_h)
            
            results['S_v'].append(S_v)
            results['E_v'].append(E_v)
            results['I_v'].append(I_v)
            results['S_h'].append(S_h)
            results['E_h'].append(E_h)
            results['I_h'].append(I_h)
            results['R_h'].append(R_h)
            results['new_cases'].append(monthly_new_cases)
            results['R0'].append(R0)
        
        return results


def create_validation_visualization(merged_df, model_results, save_path):
    """创建验证可视化"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    months = range(len(model_results['new_cases']))
    
    # 1. 实际病例 vs 模型预测
    ax1 = axes[0, 0]
    actual_cases = merged_df['cases'].values[:len(months)]
    predicted_cases = np.array(model_results['new_cases'])
    
    # 归一化比较
    if actual_cases.max() > 0:
        scale_factor = actual_cases.max() / max(predicted_cases.max(), 1)
        predicted_scaled = predicted_cases * scale_factor
    else:
        predicted_scaled = predicted_cases
    
    ax1.plot(months, actual_cases, 'b-', linewidth=2, label='Actual Cases', marker='o', markersize=3)
    ax1.plot(months, predicted_scaled, 'r--', linewidth=2, label='Model Predicted (scaled)', marker='s', markersize=3)
    ax1.set_xlabel('Month Index')
    ax1.set_ylabel('Cases')
    ax1.set_title('Dengue Cases: Actual vs Model Prediction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. BI预测 vs 实际
    ax2 = axes[0, 1]
    actual_bi = merged_df['bi'].values[:len(months)]
    valid_bi_mask = ~np.isnan(actual_bi)
    
    if valid_bi_mask.any():
        valid_indices = np.where(valid_bi_mask)[0]
        ax2.scatter(valid_indices, actual_bi[valid_bi_mask], c='blue', s=50, label='Actual BI', zorder=5)
    
    # 使用lambda_v估算BI
    if 'lambda_v' in model_results:
        estimated_bi = np.array(model_results['lambda_v']) / 400  # 简单转换
        ax2.plot(months, estimated_bi, 'r-', linewidth=2, label='Estimated BI from Model')
    
    ax2.set_xlabel('Month Index')
    ax2.set_ylabel('Breteau Index')
    ax2.set_title('Breteau Index Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. R0时间序列
    ax3 = axes[1, 0]
    r0_values = model_results['R0']
    ax3.plot(months, r0_values, 'g-', linewidth=2)
    ax3.axhline(y=1, color='r', linestyle='--', label='R0=1 threshold')
    ax3.fill_between(months, 0, r0_values, where=np.array(r0_values) > 1, 
                     color='red', alpha=0.3, label='Epidemic (R0>1)')
    ax3.set_xlabel('Month Index')
    ax3.set_ylabel('R0')
    ax3.set_title('Basic Reproduction Number (R0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 人群仓室动态
    ax4 = axes[1, 1]
    ax4.plot(months, np.array(model_results['E_h'])/1e6, 'orange', linewidth=2, label='Exposed (E_h)')
    ax4.plot(months, np.array(model_results['I_h'])/1e6, 'red', linewidth=2, label='Infectious (I_h)')
    ax4.plot(months, np.array(model_results['R_h'])/1e6, 'green', linewidth=2, label='Recovered (R_h)')
    ax4.set_xlabel('Month Index')
    ax4.set_ylabel('Population (millions)')
    ax4.set_title('Human Population Dynamics (SEI-SEIR)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 蚊虫仓室动态
    ax5 = axes[2, 0]
    ax5.plot(months, np.array(model_results['S_v'])/1e6, 'blue', linewidth=2, label='Susceptible (S_v)')
    ax5.plot(months, np.array(model_results['E_v'])/1e6, 'orange', linewidth=2, label='Exposed (E_v)')
    ax5.plot(months, np.array(model_results['I_v'])/1e6, 'red', linewidth=2, label='Infectious (I_v)')
    ax5.set_xlabel('Month Index')
    ax5.set_ylabel('Mosquito Population (millions)')
    ax5.set_title('Mosquito Population Dynamics (SEI)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 病例与R0相关性
    ax6 = axes[2, 1]
    ax6.scatter(r0_values, actual_cases, c='blue', alpha=0.6, s=30)
    ax6.set_xlabel('R0')
    ax6.set_ylabel('Actual Cases')
    ax6.set_title('Correlation: R0 vs Actual Cases')
    ax6.grid(True, alpha=0.3)
    
    # 计算相关系数
    valid_mask = ~np.isnan(actual_cases)
    if valid_mask.sum() > 2:
        corr = np.corrcoef(np.array(r0_values)[valid_mask], actual_cases[valid_mask])[0, 1]
        ax6.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax6.transAxes, 
                fontsize=12, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig


def main():
    print("=" * 60)
    print("SEI-SEIR模型人群动态验证 - 使用广东省实际病例数据")
    print("=" * 60)
    
    # 加载数据
    print("\n[1] 加载数据...")
    case_df = load_case_data('../data/guangdong_dengue_cases.csv')
    print(f"  病例数据: {len(case_df)} 个月 ({case_df['date'].min()} 至 {case_df['date'].max()})")
    print(f"  总病例数: {case_df['cases'].sum():,}")
    
    bi_df = load_bi_data('../data/BI.csv')
    print(f"  BI数据: {len(bi_df)} 条记录")
    
    # 合并数据
    merged_df = merge_data(case_df, bi_df)
    print(f"  合并后数据: {len(merged_df)} 条")
    
    # 筛选有BI数据的时间段 (2006-2014)
    analysis_df = merged_df[(merged_df['year'] >= 2006) & (merged_df['year'] <= 2014)].copy()
    print(f"  分析时段 (2006-2014): {len(analysis_df)} 个月")
    
    # 初始化模型
    print("\n[2] 初始化SEI-SEIR模型...")
    simulator = ImprovedSEISEIRSimulator(N_h=110000000, N_v_ratio=2.0)
    
    # 计算lambda_v序列
    print("\n[3] 计算蚊虫动态参数...")
    lambda_v_series = []
    for _, row in analysis_df.iterrows():
        lambda_v = simulator.compute_lambda_v(
            row['temperature'], 
            row['humidity'], 
            row['precipitation']
        )
        lambda_v_series.append(lambda_v)
    
    # 运行模拟
    print("\n[4] 运行SEI-SEIR模拟...")
    # 使用2006年7月的初始感染数
    I_h_init = analysis_df.iloc[0]['cases'] if analysis_df.iloc[0]['cases'] > 0 else 10
    
    results = simulator.simulate(
        months=len(analysis_df),
        weather_data=analysis_df,
        lambda_v_series=lambda_v_series,
        I_h_init=I_h_init
    )
    results['lambda_v'] = lambda_v_series
    
    # 评估结果
    print("\n[5] 模型评估...")
    actual_cases = analysis_df['cases'].values
    predicted_cases = np.array(results['new_cases'])
    r0_values = np.array(results['R0'])
    
    # 计算相关性
    corr_r0_cases = np.corrcoef(r0_values, actual_cases)[0, 1]
    print(f"  R0与实际病例相关系数: {corr_r0_cases:.4f}")
    
    # R0统计
    print(f"\n  R0统计:")
    print(f"    均值: {np.mean(r0_values):.3f}")
    print(f"    最大值: {np.max(r0_values):.3f}")
    print(f"    R0>1的月份: {(r0_values > 1).sum()} / {len(r0_values)}")
    
    # 2014年暴发分析
    df_2014 = analysis_df[analysis_df['year'] == 2014]
    idx_2014 = analysis_df[analysis_df['year'] == 2014].index - analysis_df.index[0]
    
    print(f"\n  2014年暴发分析:")
    print(f"    实际病例峰值: {df_2014['cases'].max():,}")
    print(f"    模型R0峰值: {r0_values[idx_2014].max():.3f}")
    
    # 可视化
    print("\n[6] 生成验证可视化...")
    save_path = '../results/case_validation_results.png'
    create_validation_visualization(analysis_df, results, save_path)
    print(f"  已保存: {save_path}")
    
    # 保存详细结果
    results_df = pd.DataFrame({
        'year': analysis_df['year'].values,
        'month': analysis_df['month'].values,
        'actual_cases': actual_cases,
        'predicted_cases': predicted_cases,
        'R0': r0_values,
        'actual_bi': analysis_df['bi'].values,
        'lambda_v': lambda_v_series,
        'I_h': results['I_h'],
        'E_h': results['E_h'],
        'R_h': results['R_h']
    })
    results_df.to_csv('../results/case_validation_predictions.csv', index=False)
    print(f"  已保存: results/case_validation_predictions.csv")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
