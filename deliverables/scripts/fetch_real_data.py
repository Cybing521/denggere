"""
获取真实数据模块

数据来源：
1. 气象数据：Open-Meteo API (免费历史气象数据)
2. BI数据：已有的CCM14数据集
3. 病例数据：模拟生成（实际需从CDC获取）

Open-Meteo API: https://open-meteo.com/
- 免费使用
- 无需API key
- 提供历史气象数据回溯至1940年
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time


def fetch_guangzhou_weather_openmeteo(start_date: str, end_date: str) -> pd.DataFrame:
    """
    从Open-Meteo API获取广州历史气象数据
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        
    Returns:
        月度气象数据DataFrame
    """
    # 广州坐标
    latitude = 23.13
    longitude = 113.26
    
    # Open-Meteo历史天气API
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "precipitation_sum"
        ],
        "timezone": "Asia/Shanghai"
    }
    
    print(f"正在从Open-Meteo获取广州气象数据 ({start_date} 至 {end_date})...")
    
    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # 解析数据
        daily = data.get("daily", {})
        
        df = pd.DataFrame({
            "date": pd.to_datetime(daily.get("time", [])),
            "temperature_mean": daily.get("temperature_2m_mean", []),
            "temperature_max": daily.get("temperature_2m_max", []),
            "temperature_min": daily.get("temperature_2m_min", []),
            "humidity": daily.get("relative_humidity_2m_mean", []),
            "precipitation": daily.get("precipitation_sum", [])
        })
        
        print(f"  获取到 {len(df)} 天的数据")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"API请求失败: {e}")
        return None


def aggregate_to_monthly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    将日度数据聚合为月度数据
    
    Args:
        daily_df: 日度数据
        
    Returns:
        月度数据
    """
    if daily_df is None or daily_df.empty:
        return None
    
    df = daily_df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # 聚合
    monthly = df.groupby(['year', 'month']).agg({
        'temperature_mean': 'mean',
        'temperature_max': 'mean',
        'temperature_min': 'mean',
        'humidity': 'mean',
        'precipitation': 'sum'  # 月累计降水
    }).reset_index()
    
    # 重命名
    monthly.columns = ['year', 'month', 'temperature', 'temp_max', 
                       'temp_min', 'humidity', 'rainfall']
    
    # 创建日期
    monthly['date'] = pd.to_datetime(
        monthly['year'].astype(str) + '-' + monthly['month'].astype(str) + '-01'
    )
    
    return monthly


def load_and_process_bi_data(bi_path: str, site: str = "Guangzhou") -> pd.DataFrame:
    """
    加载并处理布雷图指数数据
    
    Args:
        bi_path: BI.csv文件路径
        site: 城市名称
        
    Returns:
        处理后的月度BI数据
    """
    print(f"\n加载 {site} 的布雷图指数数据...")
    
    try:
        df = pd.read_csv(bi_path, encoding='latin-1')
    except:
        df = pd.read_csv(bi_path, encoding='utf-8')
    
    # 筛选指定城市的Breteau index数据
    bi_data = df[
        (df['Site_L2'] == site) &
        (df['Site_method'] == 'Breteau index') &
        (df['Site_month'] != 'Total')
    ].copy()
    
    # 处理数据
    bi_data['year'] = pd.to_numeric(bi_data['Site_year'], errors='coerce')
    bi_data['month'] = pd.to_numeric(bi_data['Site_month'], errors='coerce')
    bi_data['BI'] = pd.to_numeric(bi_data['Den_admin'].fillna(bi_data['Den_hab']), errors='coerce')
    
    # 去除无效数据
    bi_data = bi_data.dropna(subset=['year', 'month', 'BI'])
    
    # 按年月聚合
    monthly_bi = bi_data.groupby(['year', 'month'])['BI'].mean().reset_index()
    monthly_bi['year'] = monthly_bi['year'].astype(int)
    monthly_bi['month'] = monthly_bi['month'].astype(int)
    
    # 创建日期
    monthly_bi['date'] = pd.to_datetime(
        monthly_bi['year'].astype(str) + '-' + monthly_bi['month'].astype(str) + '-01'
    )
    
    print(f"  找到 {len(monthly_bi)} 条月度BI记录")
    print(f"  时间范围: {monthly_bi['year'].min()}-{monthly_bi['year'].max()}")
    
    return monthly_bi


def merge_weather_and_bi(weather_df: pd.DataFrame, bi_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并气象数据和BI数据
    
    Args:
        weather_df: 月度气象数据
        bi_df: 月度BI数据
        
    Returns:
        合并后的数据
    """
    merged = pd.merge(
        weather_df,
        bi_df[['year', 'month', 'BI']],
        on=['year', 'month'],
        how='left'
    )
    
    return merged


def generate_simulated_cases(bi_series: np.ndarray, 
                             population: int = 14_000_000,
                             lag: int = 1) -> np.ndarray:
    """
    基于BI生成模拟登革热病例数据
    
    这是基于流行病学关系的模拟数据，实际数据需要从CDC获取
    
    Args:
        bi_series: BI时间序列
        population: 人口数
        lag: BI与病例的时滞（月）
        
    Returns:
        模拟病例数
    """
    # 病例与BI的非线性关系（基于文献）
    # 当BI > 20时，登革热暴发风险显著增加
    
    # 添加时滞
    bi_lagged = np.roll(bi_series, lag)
    bi_lagged[:lag] = bi_series[:lag]
    
    # 风险函数
    risk = 1 / (1 + np.exp(-0.15 * (bi_lagged - 15)))
    
    # 基础发病率（每百万人）
    base_rate = 0.5  # 每月每百万人0.5例基础发病
    
    # 期望病例数
    expected = population / 1e6 * base_rate * (1 + 50 * risk)
    
    # 泊松随机
    np.random.seed(42)
    cases = np.random.poisson(expected)
    
    return cases


class RealDataFetcher:
    """真实数据获取器"""
    
    def __init__(self, bi_path: str, output_dir: str = "/root/wenmei/data/real"):
        self.bi_path = bi_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.weather_data = None
        self.bi_data = None
        self.merged_data = None
        
    def fetch_all(self, start_year: int = 2006, end_year: int = 2014):
        """获取所有数据"""
        print("=" * 60)
        print("获取真实数据")
        print("=" * 60)
        
        # 1. 获取气象数据
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        daily_weather = fetch_guangzhou_weather_openmeteo(start_date, end_date)
        
        if daily_weather is not None:
            self.weather_data = aggregate_to_monthly(daily_weather)
            print(f"\n气象数据聚合完成: {len(self.weather_data)} 个月")
        else:
            print("警告: 无法获取气象数据，将使用备用方法")
            self.weather_data = self._generate_fallback_weather(start_year, end_year)
        
        # 2. 加载BI数据
        self.bi_data = load_and_process_bi_data(self.bi_path, "Guangzhou")
        
        # 3. 合并数据
        print("\n合并数据...")
        self.merged_data = merge_weather_and_bi(self.weather_data, self.bi_data)
        
        # 4. 统计有效数据
        valid_bi = self.merged_data['BI'].notna().sum()
        print(f"  总月份数: {len(self.merged_data)}")
        print(f"  有效BI数据: {valid_bi} 个月")
        
        # 5. 生成模拟病例数据
        bi_interp = self.merged_data['BI'].interpolate(method='linear', limit_direction='both').values
        self.merged_data['cases_simulated'] = generate_simulated_cases(bi_interp)
        
        return self
    
    def _generate_fallback_weather(self, start_year: int, end_year: int) -> pd.DataFrame:
        """生成备用气象数据（基于气候学平均值）"""
        print("使用广州气候学平均值作为备用...")
        
        # 广州月度气候特征（基于历史统计）
        climate = {
            1: {'temp': 13.6, 'humidity': 68, 'rainfall': 42},
            2: {'temp': 14.5, 'humidity': 73, 'rainfall': 68},
            3: {'temp': 17.8, 'humidity': 79, 'rainfall': 94},
            4: {'temp': 22.0, 'humidity': 83, 'rainfall': 188},
            5: {'temp': 25.7, 'humidity': 83, 'rainfall': 284},
            6: {'temp': 27.6, 'humidity': 83, 'rainfall': 276},
            7: {'temp': 28.6, 'humidity': 81, 'rainfall': 228},
            8: {'temp': 28.4, 'humidity': 81, 'rainfall': 221},
            9: {'temp': 27.0, 'humidity': 77, 'rainfall': 172},
            10: {'temp': 24.0, 'humidity': 71, 'rainfall': 79},
            11: {'temp': 19.5, 'humidity': 66, 'rainfall': 36},
            12: {'temp': 15.3, 'humidity': 63, 'rainfall': 30}
        }
        
        data = []
        np.random.seed(42)
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                c = climate[month]
                data.append({
                    'year': year,
                    'month': month,
                    'temperature': c['temp'] + np.random.normal(0, 1.5),
                    'temp_max': c['temp'] + 5 + np.random.normal(0, 1),
                    'temp_min': c['temp'] - 5 + np.random.normal(0, 1),
                    'humidity': np.clip(c['humidity'] + np.random.normal(0, 5), 40, 100),
                    'rainfall': max(0, c['rainfall'] * np.random.lognormal(0, 0.3))
                })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')
        
        return df
    
    def save(self):
        """保存数据"""
        if self.weather_data is not None:
            path = self.output_dir / "guangzhou_weather_real.csv"
            self.weather_data.to_csv(path, index=False)
            print(f"\n气象数据已保存: {path}")
        
        if self.bi_data is not None:
            path = self.output_dir / "guangzhou_bi.csv"
            self.bi_data.to_csv(path, index=False)
            print(f"BI数据已保存: {path}")
        
        if self.merged_data is not None:
            path = self.output_dir / "guangzhou_merged_real.csv"
            self.merged_data.to_csv(path, index=False)
            print(f"合并数据已保存: {path}")
    
    def get_summary(self) -> dict:
        """获取数据摘要"""
        if self.merged_data is None:
            return {}
        
        bi_valid = self.merged_data['BI'].dropna()
        
        return {
            'weather_source': 'Open-Meteo API' if hasattr(self, '_api_success') else 'Climate Average',
            'years': f"{self.merged_data['year'].min()}-{self.merged_data['year'].max()}",
            'total_months': len(self.merged_data),
            'valid_bi_months': len(bi_valid),
            'bi_mean': bi_valid.mean() if len(bi_valid) > 0 else 0,
            'bi_std': bi_valid.std() if len(bi_valid) > 0 else 0,
            'bi_max': bi_valid.max() if len(bi_valid) > 0 else 0,
            'temp_mean': self.merged_data['temperature'].mean(),
            'rainfall_total': self.merged_data['rainfall'].sum()
        }


def main():
    """主函数"""
    # 配置
    bi_path = "/root/wenmei/data/BI.csv"
    
    # 创建数据获取器
    fetcher = RealDataFetcher(bi_path)
    
    # 获取数据
    fetcher.fetch_all(start_year=2006, end_year=2014)
    
    # 保存数据
    fetcher.save()
    
    # 打印摘要
    summary = fetcher.get_summary()
    print("\n" + "=" * 60)
    print("数据摘要")
    print("=" * 60)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 显示数据预览
    print("\n合并数据预览:")
    print(fetcher.merged_data[['year', 'month', 'temperature', 'humidity', 
                               'rainfall', 'BI', 'cases_simulated']].head(20).to_string())
    
    return fetcher


if __name__ == "__main__":
    fetcher = main()
