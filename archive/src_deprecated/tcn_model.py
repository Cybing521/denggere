"""
TCN (Temporal Convolutional Network) 模型
用于从气象数据预测蚊虫承载力 Λ_v(t)

架构:
    Input: [温度, 湿度, 降雨量] × 历史窗口
    ↓
    TCN层 (膨胀因果卷积)
    ↓
    全连接层
    ↓
    Output: Λ_v(t) 或 预测的BI值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional


class CausalConv1d(nn.Module):
    """因果卷积层 - 只使用过去的信息"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        out = self.conv(x)
        # 移除未来的padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """TCN的基本单元"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        
        # 第一层卷积
        self.conv1 = CausalConv1d(in_channels, out_channels, 
                                  kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二层卷积
        self.conv2 = CausalConv1d(out_channels, out_channels,
                                  kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network
    用于气象数据 → 蚊虫承载力预测
    """
    
    def __init__(self, input_size: int, output_size: int,
                 num_channels: List[int] = [32, 64, 64],
                 kernel_size: int = 3, dropout: float = 0.2):
        """
        Args:
            input_size: 输入特征维度 (如: 温度、湿度、降雨 = 3)
            output_size: 输出维度 (如: 预测BI = 1)
            num_channels: 每层的通道数
            kernel_size: 卷积核大小
            dropout: dropout率
        """
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size,
                                       dilation, dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_size]
        Returns:
            [batch_size, output_size]
        """
        # TCN期望输入为 [batch, channels, seq_len]
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # 取最后一个时间步
        out = out[:, :, -1]
        out = self.linear(out)
        return out


class MosquitoDataset(Dataset):
    """蚊虫数据集"""
    
    def __init__(self, weather_data: np.ndarray, bi_data: np.ndarray,
                 seq_length: int = 12):
        """
        Args:
            weather_data: 气象数据 [n_samples, n_features]
            bi_data: 布雷图指数 [n_samples]
            seq_length: 历史窗口长度（月）
        """
        self.weather = torch.FloatTensor(weather_data)
        self.bi = torch.FloatTensor(bi_data)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.bi) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.weather[idx:idx+self.seq_length]
        y = self.bi[idx+self.seq_length]
        return x, y


class MosquitoTCNPredictor:
    """
    TCN预测器 - 用于预测蚊虫承载力/布雷图指数
    """
    
    def __init__(self, input_size: int = 3, hidden_channels: List[int] = [32, 64, 64],
                 seq_length: int = 6, learning_rate: float = 0.001,
                 device: str = None):
        """
        Args:
            input_size: 输入特征数（温度、湿度、降雨等）
            hidden_channels: TCN隐藏层通道数
            seq_length: 历史窗口长度
            learning_rate: 学习率
            device: 计算设备
        """
        self.seq_length = seq_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = TCN(
            input_size=input_size,
            output_size=1,
            num_channels=hidden_channels,
            kernel_size=3,
            dropout=0.2
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 数据标准化参数
        self.weather_mean = None
        self.weather_std = None
        self.bi_mean = None
        self.bi_std = None
        
    def _normalize(self, weather: np.ndarray, bi: np.ndarray, fit: bool = True):
        """数据标准化"""
        if fit:
            self.weather_mean = weather.mean(axis=0)
            self.weather_std = weather.std(axis=0) + 1e-8
            self.bi_mean = bi.mean()
            self.bi_std = bi.std() + 1e-8
            
        weather_norm = (weather - self.weather_mean) / self.weather_std
        bi_norm = (bi - self.bi_mean) / self.bi_std
        
        return weather_norm, bi_norm
    
    def _denormalize_bi(self, bi_norm: np.ndarray) -> np.ndarray:
        """反标准化BI"""
        return bi_norm * self.bi_std + self.bi_mean
    
    def train(self, weather_data: np.ndarray, bi_data: np.ndarray,
              epochs: int = 100, batch_size: int = 16, 
              validation_split: float = 0.2, verbose: bool = True):
        """
        训练模型
        
        Args:
            weather_data: 气象数据 [n_samples, n_features]
            bi_data: 布雷图指数 [n_samples]
            epochs: 训练轮数
            batch_size: 批次大小
            validation_split: 验证集比例
            verbose: 是否打印训练过程
            
        Returns:
            训练历史 (train_loss, val_loss)
        """
        # 数据标准化
        weather_norm, bi_norm = self._normalize(weather_data, bi_data, fit=True)
        
        # 划分训练集和验证集
        n = len(bi_norm)
        n_train = int(n * (1 - validation_split))
        
        train_dataset = MosquitoDataset(
            weather_norm[:n_train], bi_norm[:n_train], self.seq_length
        )
        val_dataset = MosquitoDataset(
            weather_norm[n_train-self.seq_length:], 
            bi_norm[n_train-self.seq_length:], 
            self.seq_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                pred = self.model(x).squeeze()
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    pred = self.model(x).squeeze()
                    val_loss += self.criterion(pred, y).item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        return history
    
    def predict(self, weather_data: np.ndarray) -> np.ndarray:
        """
        预测布雷图指数
        
        Args:
            weather_data: 气象数据序列 [seq_length, n_features]
            
        Returns:
            预测的BI值
        """
        self.model.eval()
        
        # 标准化
        weather_norm = (weather_data - self.weather_mean) / self.weather_std
        
        with torch.no_grad():
            x = torch.FloatTensor(weather_norm).unsqueeze(0).to(self.device)
            pred_norm = self.model(x).squeeze().cpu().numpy()
            pred = self._denormalize_bi(pred_norm)
        
        return pred
    
    def predict_sequence(self, weather_sequence: np.ndarray) -> np.ndarray:
        """
        预测BI序列
        
        Args:
            weather_sequence: 气象数据序列 [n_samples, n_features]
            
        Returns:
            预测的BI序列
        """
        predictions = []
        
        for i in range(self.seq_length, len(weather_sequence)):
            weather_window = weather_sequence[i-self.seq_length:i]
            pred = self.predict(weather_window)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_lambda_v_function(self, weather_data: np.ndarray,
                               base_density: float = 1000) -> callable:
        """
        获取用于SEI-SEIR模型的 Λ_v(t) 函数
        
        Args:
            weather_data: 完整的气象数据序列
            base_density: 基础蚊虫密度
            
        Returns:
            Λ_v(t) 函数
        """
        # 预测BI序列
        bi_pred = self.predict_sequence(weather_data)
        
        # 创建插值函数
        from scipy.interpolate import interp1d
        t = np.arange(len(bi_pred))
        
        def lambda_v(time):
            # 将BI转换为蚊虫出生率
            if time < 0:
                bi = bi_pred[0]
            elif time >= len(bi_pred):
                bi = bi_pred[-1]
            else:
                bi = np.interp(time, t, bi_pred)
            
            # BI → Λ_v 转换
            lambda_val = base_density * (1 - np.exp(-0.05 * bi))
            return max(100, lambda_val)  # 确保最小值
        
        return lambda_v
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'weather_mean': self.weather_mean,
            'weather_std': self.weather_std,
            'bi_mean': self.bi_mean,
            'bi_std': self.bi_std,
            'seq_length': self.seq_length
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.weather_mean = checkpoint['weather_mean']
        self.weather_std = checkpoint['weather_std']
        self.bi_mean = checkpoint['bi_mean']
        self.bi_std = checkpoint['bi_std']
        self.seq_length = checkpoint['seq_length']


if __name__ == "__main__":
    # 测试TCN模型
    print("测试TCN模型...")
    
    # 创建模拟数据
    n_samples = 120  # 10年月度数据
    
    # 模拟气象数据（温度、湿度、降雨）
    np.random.seed(42)
    months = np.arange(n_samples)
    
    # 广州气候特征的模拟
    temperature = 22 + 8 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 2, n_samples)
    humidity = 75 + 10 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 5, n_samples)
    rainfall = 100 + 80 * np.sin(2 * np.pi * (months - 3) / 12) + np.random.exponential(20, n_samples)
    
    weather_data = np.column_stack([temperature, humidity, rainfall])
    
    # 模拟BI数据（与气象相关）
    bi = 5 + 0.3 * temperature + 0.05 * humidity + 0.02 * rainfall + np.random.normal(0, 2, n_samples)
    bi = np.maximum(0, bi)
    
    # 创建预测器
    predictor = MosquitoTCNPredictor(
        input_size=3,
        hidden_channels=[32, 64, 32],
        seq_length=6
    )
    
    # 训练
    print("\n开始训练...")
    history = predictor.train(
        weather_data, bi,
        epochs=50,
        batch_size=8,
        verbose=True
    )
    
    # 预测
    predictions = predictor.predict_sequence(weather_data)
    
    # 评估
    actual = bi[6:]  # 跳过seq_length
    mse = np.mean((predictions - actual)**2)
    r2 = 1 - np.sum((predictions - actual)**2) / np.sum((actual - actual.mean())**2)
    
    print(f"\n预测结果:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # 可视化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 训练损失
    axes[0, 0].plot(history['train_loss'], label='训练损失')
    axes[0, 0].plot(history['val_loss'], label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练过程')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 预测对比
    axes[0, 1].plot(actual, label='实际BI', alpha=0.7)
    axes[0, 1].plot(predictions, label='预测BI', alpha=0.7)
    axes[0, 1].set_xlabel('月份')
    axes[0, 1].set_ylabel('布雷图指数')
    axes[0, 1].set_title('BI预测对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 气象数据
    axes[1, 0].plot(temperature, label='温度 (°C)')
    axes[1, 0].plot(humidity, label='湿度 (%)')
    axes[1, 0].plot(rainfall/10, label='降雨/10 (mm)')
    axes[1, 0].set_xlabel('月份')
    axes[1, 0].set_title('气象数据')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 散点图
    axes[1, 1].scatter(actual, predictions, alpha=0.5)
    axes[1, 1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
                    'r--', label='y=x')
    axes[1, 1].set_xlabel('实际BI')
    axes[1, 1].set_ylabel('预测BI')
    axes[1, 1].set_title(f'预测 vs 实际 (R²={r2:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('/root/wenmei/results/tcn_test.png', dpi=150)
    plt.close()
    
    print("\nTCN模型测试完成!")
