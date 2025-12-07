#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision.models as models
import math

class MultimodalAttention(nn.Module):
    """多模态注意力融合模块，用于融合不同模态的特征"""
    
    def __init__(self, visual_dim, traffic_dim, vehicle_dim, appearance_dim, attributes_dim, hidden_dim=128):
        """
        初始化多模态注意力融合模块
        
        Args:
            visual_dim (int): 视觉特征维度
            traffic_dim (int): 交通场景特征维度
            vehicle_dim (int): 车辆行为特征维度
            appearance_dim (int): 行人外观特征维度
            attributes_dim (int): 行人属性特征维度
            hidden_dim (int): 隐藏层维度
        """
        super(MultimodalAttention, self).__init__()
        
        # 各模态特征投影层
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.traffic_proj = nn.Linear(traffic_dim, hidden_dim)
        self.vehicle_proj = nn.Linear(vehicle_dim, hidden_dim)
        self.appearance_proj = nn.Linear(appearance_dim, hidden_dim)
        self.attributes_proj = nn.Linear(attributes_dim, hidden_dim)
        
        # 注意力层
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # LayerNorm层
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, visual_features, traffic_features, vehicle_features, appearance_features, attributes_features):
        """
        前向传播
        
        Args:
            visual_features (Tensor): 视觉特征 [batch_size, visual_dim]
            traffic_features (Tensor): 交通场景特征 [batch_size, traffic_dim]
            vehicle_features (Tensor): 车辆行为特征 [batch_size, vehicle_dim]
            appearance_features (Tensor): 行人外观特征 [batch_size, appearance_dim]
            attributes_features (Tensor): 行人属性特征 [batch_size, attributes_dim]
        
        Returns:
            Tensor: 融合后的特征 [batch_size, hidden_dim]
        """
        # 特征投影
        visual_proj = self.visual_proj(visual_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        traffic_proj = self.traffic_proj(traffic_features).unsqueeze(1)
        vehicle_proj = self.vehicle_proj(vehicle_features).unsqueeze(1)
        appearance_proj = self.appearance_proj(appearance_features).unsqueeze(1)
        attributes_proj = self.attributes_proj(attributes_features).unsqueeze(1)
        
        # 拼接不同模态的特征
        multimodal_features = torch.cat([
            visual_proj, traffic_proj, vehicle_proj, appearance_proj, attributes_proj
        ], dim=1)  # [batch_size, 5, hidden_dim]
        
        # 自注意力机制
        attn_output, _ = self.attention(multimodal_features, multimodal_features, multimodal_features)
        
        # 残差连接和归一化
        attn_output = attn_output + multimodal_features
        attn_output = self.norm(attn_output)
        
        # 扁平化特征
        flat_features = attn_output.reshape(attn_output.size(0), -1)  # [batch_size, 5*hidden_dim]
        
        # 特征融合
        fused_features = self.fusion_layer(flat_features)  # [batch_size, hidden_dim]
        
        return fused_features

class MultimodalFusionModel(nn.Module):
    """
    多模态融合模型
    
    结合视觉、交通场景、车辆行为、行人外观和属性特征，进行行人行为预测
    """
    
    def __init__(self, num_classes=2, hidden_dim=128, dropout=0.3, sequence_length=16):
        """
        初始化多模态融合模型
        
        Args:
            num_classes (int): 类别数，默认为2（过马路/不过马路）
            hidden_dim (int): 隐藏层维度
            dropout (float): Dropout比率
            sequence_length (int): 视频序列长度
        """
        super(MultimodalFusionModel, self).__init__()
        
        # 视觉特征提取器（使用MobileNetV2）
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.visual_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # 视觉特征池化层
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 视觉特征维度
        self.visual_dim = 1280
        
        # 序列建模器（GRU）
        self.gru = nn.GRU(
            input_size=self.visual_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 特征维度定义
        self.traffic_dim = 5  # 交通场景特征维度
        self.vehicle_dim = 4  # 车辆行为特征维度
        self.appearance_dim = 8  # 行人外观特征维度
        self.attributes_dim = 6  # 行人属性特征维度
        
        # 多模态融合模块
        self.multimodal_fusion = MultimodalAttention(
            visual_dim=hidden_dim * 2,  # 双向GRU的输出维度
            traffic_dim=self.traffic_dim,
            vehicle_dim=self.vehicle_dim,
            appearance_dim=self.appearance_dim,
            attributes_dim=self.attributes_dim,
            hidden_dim=hidden_dim
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, frames, multimodal_features):
        """
        前向传播
        
        Args:
            frames (Tensor): 视频帧序列，形状为 [batch_size, sequence_length, channels, height, width]
            multimodal_features (dict): 多模态特征字典，包含交通场景、车辆行为、行人外观和属性特征
        
        Returns:
            Tensor: 预测结果，形状为 [batch_size, num_classes]
        """
        # 从多模态特征字典中提取各种特征
        traffic_features = multimodal_features['traffic']
        vehicle_features = multimodal_features['vehicle']
        appearance_features = multimodal_features['appearance']
        attributes_features = multimodal_features['attributes']
        
        # 1. 视觉时序特征提取
        batch_size, seq_len, c, h, w = frames.shape
        
        # 重塑输入以便批量处理所有帧
        frames = frames.view(batch_size * seq_len, c, h, w)
        
        # 通过视觉编码器提取特征
        visual_features = self.visual_encoder(frames)
        
        # 池化
        visual_features = self.pool(visual_features)
        
        # 扁平化
        visual_features = visual_features.view(batch_size * seq_len, -1)
        
        # 重塑为序列形式
        visual_features = visual_features.view(batch_size, seq_len, -1)
        
        # 通过GRU处理序列
        gru_output, _ = self.gru(visual_features)
        
        # 使用最后一个时间步的输出
        visual_seq_features = gru_output[:, -1, :]  # [batch_size, hidden_dim*2]
        
        # 2. 多模态特征融合
        fused_features = self.multimodal_fusion(
            visual_seq_features,
            traffic_features,
            vehicle_features,
            appearance_features,
            attributes_features
        )
        
        # 应用dropout
        fused_features = self.dropout(fused_features)
        
        # 3. 分类预测
        output = self.classifier(fused_features)
        
        return output
    
    def get_weights(self):
        """获取模型权重"""
        return {k: v.cpu() for k, v in self.state_dict().items()}
    
    def set_weights(self, weights):
        """设置模型权重"""
        self.load_state_dict(weights)
    
    def extract_visual_features(self, frames):
        """
        仅提取视觉特征，用于特征可视化或其他分析
        
        Args:
            frames (Tensor): 视频帧序列
        
        Returns:
            Tensor: 提取的视觉特征
        """
        batch_size, seq_len, c, h, w = frames.shape
        
        # 重塑输入
        frames = frames.view(batch_size * seq_len, c, h, w)
        
        # 提取特征
        features = self.visual_encoder(frames)
        features = self.pool(features)
        features = features.view(batch_size, seq_len, -1)
        
        return features

def create_model(device, **kwargs):
    """
    创建多模态融合模型实例
    
    Args:
        device: 训练设备（CPU/GPU）
        **kwargs: 其他参数
    
    Returns:
        MultimodalFusionModel: 模型实例
    """
    model = MultimodalFusionModel(**kwargs)
    return model.to(device) 