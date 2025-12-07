#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetLSTM(nn.Module):
    """
    MobileNetV3 + LSTM 模型用于边缘节点2
    
    使用MobileNetV3轻量级网络提取视频帧的特征，然后使用LSTM处理时序特征
    """
    
    def __init__(self, num_classes=2, hidden_size=128, num_layers=1, dropout=0.3):
        """
        初始化MobileNetLSTM模型
        
        Args:
            num_classes (int): 类别数，默认为2（过马路/不过马路）
            hidden_size (int): LSTM隐藏层大小
            num_layers (int): LSTM层数
            dropout (float): Dropout比率
        """
        super(MobileNetLSTM, self).__init__()
        
        # 加载预训练的MobileNetV3-Small模型 (而不是Large，以减少内存使用)
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        # 移除最后的分类器层
        self.mobilenet_features = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # 添加自适应平均池化层，确保输出固定大小
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征维度（MobileNetV3-Small的输出是576，比Large的960小）
        self.feature_dim = 576
        
        # LSTM层处理时序特征
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 双向LSTM，所以输出维度是hidden_size*2
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量，形状为 [batch_size, sequence_length, channels, height, width]
        
        Returns:
            Tensor: 预测结果，形状为 [batch_size, num_classes]
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # 重塑输入以便批量处理所有帧
        x = x.view(batch_size * seq_len, c, h, w)
        
        # 通过MobileNet提取特征
        x = self.mobilenet_features(x)
        
        # 应用池化层确保输出固定维度
        x = self.pool(x)
        
        # 扁平化特征
        x = x.view(batch_size * seq_len, -1)
        
        # 修复：重塑回 [batch_size, sequence_length, features]
        x = x.view(batch_size, seq_len, self.feature_dim)
        
        # 通过LSTM处理序列
        x, _ = self.lstm(x)
        
        # 使用最后一个时间步的输出
        x = x[:, -1, :]
        
        # 应用dropout
        x = self.dropout(x)
        
        # 分类层
        x = self.fc(x)
        
        return x
    
    def get_weights(self):
        """获取模型权重"""
        return {k: v.cpu() for k, v in self.state_dict().items()}
    
    def set_weights(self, weights):
        """设置模型权重"""
        self.load_state_dict(weights)

def create_model(device, **kwargs):
    """
    创建MobileNetLSTM模型实例
    
    Args:
        device: 训练设备（CPU/GPU）
        **kwargs: 其他参数
    
    Returns:
        MobileNetLSTM: 模型实例
    """
    model = MobileNetLSTM(**kwargs)
    return model.to(device) 