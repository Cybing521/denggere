"""
使用真实数据训练 SEI-SEIR + TCN 模型

数据来源：
- 气象数据：Open-Meteo API / 广州气候学平均值
- BI数据：CCM14数据集（广州2006-2023年）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from tcn_model import MosquitoTCNPredictor
from sei_seir_model import SEISEIRModel, SEISEIRParameters
from improved_model import bi_to_lambda_v, ImprovedSEISEIRSimulator

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_real_data(data_path: str = "/root/wenmei/data/real/guangzhou_merged_real.csv"):
    """加载真实数据"""
    print("加载真实数据...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  数据范围: {df['year'].min()}-{df['year'].max()}")
    print(f"  总月份数: {len(df)}")
    print(f"  有效BI数据: {df['BI'].notna().sum()}")
    
    return df


def prepare_training_data(df: pd.DataFrame, seq_length: int = 6):
    """准备训练数据"""
    # 气象特征
    weather_cols = ['temperature', 'humidity', 'rainfall']
    weather_data = df[weather_cols].values
    
    # BI数据（插值填充缺失值）
    bi_data = df['BI'].interpolate(method='linear', limit_direction='both').values
    
    # 有效掩码
    valid_mask = df['BI'].notna().values
    
    return weather_data, bi_data, valid_mask


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray, 
                         valid_mask: np.ndarray = None) -> dict:
    """评估预测性能"""
    if valid_mask is not None:
        # 只评估有效数据点
        actual_valid = actual[valid_mask]
        # 对齐预测值
        pred_indices = np.where(valid_mask)[0]
        pred_valid = []
        for idx in pred_indices:
            if idx < len(predicted):
                pred_valid.append(predicted[idx])
        pred_valid = np.array(pred_valid)
        
        if len(pred_valid) != len(actual_valid):
            min_len = min(len(pred_valid), len(actual_valid))
            pred_valid = pred_valid[:min_len]
            actual_valid = actual_valid[:min_len]
    else:
        actual_valid = actual
        pred_valid = predicted[:len(actual)]
    
    if len(actual_valid) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'n_samples': 0}
    
    mae = np.mean(np.abs(pred_valid - actual_valid))
    rmse = np.sqrt(np.mean((pred_valid - actual_valid)**2))
    
    ss_res = np.sum((pred_valid - actual_valid)**2)
    ss_tot = np.sum((actual_valid - actual_valid.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # 相关系数
    if len(actual_valid) > 1:
        corr = np.corrcoef(actual_valid, pred_valid)[0, 1]
    else:
        corr = np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'correlation': corr,
        'n_samples': len(actual_valid)
    }


def create_comprehensive_figure(df: pd.DataFrame, bi_pred: np.ndarray,
                                 sei_seir_result: dict, r0_series: np.ndarray,
                                 metrics: dict, history: dict,
                                 save_path: str):
    """创建综合可视化图"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. TCN训练曲线
    ax1 = fig.add_subplot(4, 3, 1)
    ax1.plot(history['train_loss'], label='Train Loss', alpha=0.8)
    ax1.plot(history['val_loss'], label='Val Loss', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('TCN Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. BI预测对比
    ax2 = fig.add_subplot(4, 3, 2)
    actual_bi = df['BI'].values
    time_idx = np.arange(len(actual_bi))
    valid_mask = ~np.isnan(actual_bi)
    
    ax2.scatter(time_idx[valid_mask], actual_bi[valid_mask], 
               color='blue', label='Observed BI', s=40, alpha=0.7)
    
    pred_start = 6  # TCN需要6个月历史
    ax2.plot(range(pred_start, pred_start + len(bi_pred)), bi_pred, 
            'r-', label='Predicted BI', linewidth=2)
    
    # 风险线
    ax2.axhline(y=5, color='green', linestyle='--', alpha=0.4, label='Safe (BI=5)')
    ax2.axhline(y=10, color='orange', linestyle='--', alpha=0.4, label='Medium (BI=10)')
    ax2.axhline(y=20, color='red', linestyle='--', alpha=0.4, label='High (BI=20)')
    
    ax2.set_xlabel('Month Index')
    ax2.set_ylabel('Breteau Index')
    ax2.set_title(f'BI Prediction (R²={metrics["r2"]:.3f}, r={metrics["correlation"]:.3f})')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 预测散点图
    ax3 = fig.add_subplot(4, 3, 3)
    actual_valid = actual_bi[valid_mask][6:]  # 跳过前6个月
    pred_for_scatter = bi_pred[:len(actual_valid)]
    
    ax3.scatter(actual_valid, pred_for_scatter, alpha=0.6, s=50)
    max_val = max(actual_valid.max(), pred_for_scatter.max())
    ax3.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    ax3.set_xlabel('Observed BI')
    ax3.set_ylabel('Predicted BI')
    ax3.set_title('Prediction vs Observation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 气象数据
    ax4 = fig.add_subplot(4, 3, 4)
    ax4.plot(df['temperature'], label='Temperature (°C)', alpha=0.8)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Temperature (°C)')
    ax4.set_title('Guangzhou Monthly Temperature (Real Data)')
    ax4.grid(True, alpha=0.3)
    
    ax4_twin = ax4.twinx()
    ax4_twin.bar(range(len(df)), df['rainfall'], alpha=0.3, color='blue', label='Rainfall')
    ax4_twin.set_ylabel('Rainfall (mm)', color='blue')
    
    # 5. 蚊虫出生率
    ax5 = fig.add_subplot(4, 3, 5)
    lambda_v = [bi_to_lambda_v(bi) for bi in bi_pred]
    ax5.plot(lambda_v, color='green', linewidth=2)
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Λ_v (Birth Rate)')
    ax5.set_title('Estimated Mosquito Birth Rate')
    ax5.grid(True, alpha=0.3)
    
    # 6. R0时间序列
    ax6 = fig.add_subplot(4, 3, 6)
    ax6.plot(r0_series, color='purple', linewidth=2)
    ax6.axhline(y=1, color='red', linestyle='--', label='Epidemic Threshold')
    ax6.fill_between(range(len(r0_series)), 0, r0_series, 
                    where=r0_series > 1, alpha=0.3, color='red')
    ax6.set_xlabel('Month')
    ax6.set_ylabel('R₀')
    ax6.set_title(f'Basic Reproduction Number (mean={r0_series.mean():.2f})')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 蚊虫SEI动态
    ax7 = fig.add_subplot(4, 3, 7)
    t = sei_seir_result['t']
    ax7.plot(t, sei_seir_result['S_v'], label='S_v', alpha=0.8)
    ax7.plot(t, sei_seir_result['E_v'], label='E_v', alpha=0.8)
    ax7.plot(t, sei_seir_result['I_v'], label='I_v', alpha=0.8)
    ax7.set_xlabel('Days')
    ax7.set_ylabel('Mosquito Population')
    ax7.set_title('Mosquito SEI Dynamics')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 人群SEIR动态
    ax8 = fig.add_subplot(4, 3, 8)
    ax8.plot(t, sei_seir_result['E_h'], label='E_h (Exposed)', alpha=0.8)
    ax8.plot(t, sei_seir_result['I_h'], label='I_h (Infectious)', alpha=0.8)
    ax8.plot(t, sei_seir_result['R_h'], label='R_h (Recovered)', alpha=0.8)
    ax8.set_xlabel('Days')
    ax8.set_ylabel('Human Population')
    ax8.set_title('Human SEIR Dynamics')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. 月度感染蚊虫与BI对比
    ax9 = fig.add_subplot(4, 3, 9)
    days_per_month = 30
    n_months = len(sei_seir_result['I_v']) // days_per_month
    monthly_Iv = [sei_seir_result['I_v'][i*days_per_month:(i+1)*days_per_month].mean() 
                  for i in range(n_months)]
    
    ax9.plot(monthly_Iv, 'b-', label='Simulated I_v', linewidth=2)
    ax9_twin = ax9.twinx()
    ax9_twin.plot(bi_pred[:n_months], 'r--', label='Predicted BI', linewidth=2)
    
    ax9.set_xlabel('Month')
    ax9.set_ylabel('Infectious Mosquitoes', color='blue')
    ax9_twin.set_ylabel('Breteau Index', color='red')
    ax9.set_title('Simulated I_v vs Predicted BI')
    ax9.legend(loc='upper left')
    ax9_twin.legend(loc='upper right')
    ax9.grid(True, alpha=0.3)
    
    # 10. BI季节性
    ax10 = fig.add_subplot(4, 3, 10)
    monthly_bi = df.groupby('month')['BI'].agg(['mean', 'std']).fillna(0)
    ax10.bar(monthly_bi.index, monthly_bi['mean'], yerr=monthly_bi['std'],
            capsize=3, color='steelblue', alpha=0.7)
    ax10.set_xlabel('Month')
    ax10.set_ylabel('Mean BI')
    ax10.set_title('Seasonal Pattern of BI')
    ax10.set_xticks(range(1, 13))
    ax10.grid(True, alpha=0.3)
    
    # 11. 温度-BI关系
    ax11 = fig.add_subplot(4, 3, 11)
    bi_valid = df.dropna(subset=['BI'])
    ax11.scatter(bi_valid['temperature'], bi_valid['BI'], alpha=0.6)
    
    # 拟合趋势线
    z = np.polyfit(bi_valid['temperature'], bi_valid['BI'], 2)
    p = np.poly1d(z)
    temp_range = np.linspace(bi_valid['temperature'].min(), bi_valid['temperature'].max(), 100)
    ax11.plot(temp_range, p(temp_range), 'r-', label='Quadratic Fit')
    
    ax11.set_xlabel('Temperature (°C)')
    ax11.set_ylabel('Breteau Index')
    ax11.set_title('Temperature vs BI')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. 模型性能总结
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.axis('off')
    
    summary_text = f"""
    Model Performance Summary
    ========================
    
    Data Source: Open-Meteo API + CCM14 Dataset
    Location: Guangzhou, China
    Period: 2006-2014
    
    TCN Prediction:
    - MAE:  {metrics['mae']:.2f}
    - RMSE: {metrics['rmse']:.2f}
    - R²:   {metrics['r2']:.3f}
    - Correlation: {metrics['correlation']:.3f}
    - Valid Samples: {metrics['n_samples']}
    
    SEI-SEIR Simulation:
    - Mean R₀: {r0_series.mean():.2f}
    - Max R₀:  {r0_series.max():.2f}
    - Min R₀:  {r0_series.min():.2f}
    - Outbreak Risk: {(r0_series > 1).mean()*100:.1f}%
    
    Model Parameters (Literature-based):
    - μ_v = 0.05 day⁻¹ (Mosquito death rate)
    - β_v = 0.5 (Human→Mosquito transmission)
    - β_h = 0.75 (Mosquito→Human transmission)
    - σ_v = 0.1 day⁻¹ (Mosquito latent rate)
    - γ = 0.143 day⁻¹ (Recovery rate)
    """
    ax12.text(0.1, 0.95, summary_text, transform=ax12.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n综合结果图已保存: {save_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("SEI-SEIR + TCN 模型 - 使用真实数据训练")
    print("=" * 70)
    
    output_dir = Path("/root/wenmei/results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 加载真实数据
    print("\n[1/6] 加载数据...")
    df = load_real_data()
    weather_data, bi_data, valid_mask = prepare_training_data(df)
    
    print(f"\n数据统计:")
    print(f"  温度范围: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")
    print(f"  湿度范围: {df['humidity'].min():.1f} - {df['humidity'].max():.1f} %")
    print(f"  降雨范围: {df['rainfall'].min():.1f} - {df['rainfall'].max():.1f} mm")
    print(f"  BI范围: {df['BI'].min():.1f} - {df['BI'].max():.1f}")
    
    # 2. 创建并训练模拟器
    print("\n[2/6] 训练TCN模型...")
    simulator = ImprovedSEISEIRSimulator(N_h=14_000_000)
    history = simulator.train_tcn(weather_data, bi_data, seq_length=6, epochs=300)
    
    # 3. 获取预测
    print("\n[3/6] 生成预测...")
    bi_pred = simulator.bi_series
    
    # 4. 评估性能
    print("\n[4/6] 评估预测性能...")
    actual_bi = df['BI'].values[6:]  # 跳过seq_length
    metrics = evaluate_predictions(actual_bi, bi_pred, valid_mask[6:])
    
    print(f"\n预测性能:")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2']:.3f}")
    print(f"  相关系数: {metrics['correlation']:.3f}")
    print(f"  有效样本: {metrics['n_samples']}")
    
    # 5. 运行SEI-SEIR模拟
    print("\n[5/6] 运行SEI-SEIR模拟...")
    n_days = len(bi_pred) * 30
    sei_seir_result = simulator.simulate(n_days=n_days)
    
    # 计算R0序列
    r0_series = simulator.compute_r0_series()
    
    print(f"\nR0统计:")
    print(f"  均值: {r0_series.mean():.2f}")
    print(f"  最大: {r0_series.max():.2f}")
    print(f"  最小: {r0_series.min():.2f}")
    print(f"  >1的比例: {(r0_series > 1).mean()*100:.1f}%")
    
    # 6. 可视化
    print("\n[6/6] 生成可视化...")
    create_comprehensive_figure(
        df, bi_pred, sei_seir_result, r0_series,
        metrics, history,
        str(output_dir / "real_data_results.png")
    )
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'month_index': range(1, len(bi_pred) + 1),
        'year': df['year'].values[6:6+len(bi_pred)],
        'month': df['month'].values[6:6+len(bi_pred)],
        'predicted_bi': bi_pred,
        'actual_bi': df['BI'].values[6:6+len(bi_pred)],
        'lambda_v': [bi_to_lambda_v(bi) for bi in bi_pred],
        'R0': r0_series
    })
    results_df.to_csv(output_dir / "real_data_predictions.csv", index=False)
    
    # 保存模型
    simulator.tcn_predictor.save(str(output_dir / "tcn_model_real_data.pt"))
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"结果保存在: {output_dir}")
    print("=" * 70)
    
    return simulator, metrics


if __name__ == "__main__":
    simulator, metrics = main()
