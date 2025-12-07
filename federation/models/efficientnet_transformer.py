#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    """位置编码器，为Transformer提供序列位置信息"""
    
    def __init__(self, d_model, max_seq_length=100):
        """
        初始化位置编码器
        
        Args:
            d_model (int): 特征维度
            max_seq_length (int): 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为非更新参数
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        添加位置编码到输入
        
        Args:
            x (Tensor): 输入张量 [batch_size, seq_length, d_model]
        
        Returns:
            Tensor: 添加位置编码后的张量
        """
        return x + self.pe[:, :x.size(1)]

class SimpleEncoder(nn.Module):
    """简化版Transformer编码器，减少内存使用"""
    
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(SimpleEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
    
    def forward(self, src):
        # 自注意力层
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class EfficientNetTransformer(nn.Module):
    """
    EfficientNet + Transformer 模型用于边缘节点3
    
    使用EfficientNet提取视频帧的特征，然后使用Transformer处理时序特征
    """
    
    def __init__(self, num_classes=2, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        """
        初始化EfficientNetTransformer模型
        
        Args:
            num_classes (int): 类别数，默认为2（过马路/不过马路）
            d_model (int): Transformer模型维度
            nhead (int): Transformer多头注意力的头数
            num_layers (int): Transformer编码器层数
            dropout (float): Dropout比率
        """
        super(EfficientNetTransformer, self).__init__()
        
        # 使用MobileNetV2替代EfficientNet减少内存使用
        mobilenet = models.mobilenet_v2(pretrained=True)
        # 移除最后的分类器层，仅保留特征提取器部分
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # 添加自适应平均池化层，确保输出固定大小
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV2的特征维度是1280
        self.feature_dim = 1280
        
        # 将特征维度投影到Transformer所需的d_model维度
        self.projection = nn.Linear(self.feature_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 使用简化版的编码器堆叠
        self.encoders = nn.ModuleList([
            SimpleEncoder(d_model, nhead, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # 分类头
        self.fc = nn.Linear(d_model, num_classes)
        
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
        
        # 通过特征提取器
        x = self.features(x)
        
        # 添加池化层确保输出固定维度
        x = self.pool(x)
        
        # 压缩维度并扁平化
        x = x.view(batch_size * seq_len, -1)
        
        # 重塑回 [batch_size, sequence_length, features]
        x = x.view(batch_size, seq_len, self.feature_dim)
        
        # 投影到Transformer所需的维度
        x = self.projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 通过编码器层
        for encoder in self.encoders:
            x = encoder(x)
        
        # 使用最后一个时间步的输出
        x = x[:, -1]
        
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
    创建EfficientNetTransformer模型实例
    
    Args:
        device: 训练设备（CPU/GPU）
        **kwargs: 其他参数
    
    Returns:
        EfficientNetTransformer: 模型实例
    """
    model = EfficientNetTransformer(**kwargs)
    return model.to(device) 