#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from torch.distributions import Normal

class AdvancedSituationAwareness:
    """
    高级态势感知模块 - 基于贝叶斯理论和深度学习
    
    实现了多模态数据融合、动态权重调整和预测结果融合
    """
    
    def __init__(self, num_nodes=4, modalities=None, visualize=False):
        """
        初始化高级态势感知模块
        
        Args:
            num_nodes (int): 边缘节点数量
            modalities (list): 模态列表，如 ['visual', 'traffic', 'vehicle', 'appearance', 'attributes']
            visualize (bool): 是否生成可视化图表
        """
        self.num_nodes = num_nodes
        self.visualize = visualize
        
        # 初始化模态列表
        self.modalities = modalities if modalities else ['visual', 'traffic', 'vehicle', 'appearance', 'attributes']
        self.num_modalities = len(self.modalities)
        
        # 节点状态
        self.node_status = [True] * num_nodes
        
        # 存储性能指标历史数据
        self.metrics_history = {
            'accuracy': [[] for _ in range(num_nodes)],
            'cpu_usage': [[] for _ in range(num_nodes)],
            'memory_usage': [[] for _ in range(num_nodes)],
            'latency': [[] for _ in range(num_nodes)],
            'timestamps': [],
            'uncertainty': [[] for _ in range(num_nodes)],
            'model_weights': [[] for _ in range(num_nodes)],
            'global_acc': [],
            'global_loss': [],
            'epochs': []
        }
        
        # 多模态融合权重 - 初始设为均等
        self.modality_weights = {
            node_id: np.ones(self.num_modalities) / self.num_modalities 
            for node_id in range(num_nodes)
        }
        
        # 节点模型权重 (FedAvg聚合权重)
        self.node_weights = np.ones(num_nodes) / num_nodes
        
        # 创建输出目录
        self.output_dir = "./output/visualization/advanced"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"高级态势感知模块初始化完成，监控{num_nodes}个边缘节点，{self.num_modalities}种模态")
        print(f"注意: 假设节点{num_nodes-1}是多模态融合节点")
    
    def _get_kernel_values(self, data, kernel_width=1.0):
        """
        计算核函数值，基于图片中的公式 μ_ij(z_ij(t))
        
        Args:
            data (numpy.ndarray): 观测数据
            kernel_width (float): 核函数宽度
            
        Returns:
            numpy.ndarray: 核函数值
        """
        # 使用高斯核函数
        return np.exp(-0.5 * (data**2) / (kernel_width**2))
    
    def fuse_multimodal_data(self, multimodal_data, weights=None, node_id=None):
        """
        多模态数据融合，基于图片中的z(t)公式
        
        Args:
            multimodal_data (dict): 多模态数据字典，格式为 {modality: data_tensor}
            weights (numpy.ndarray, optional): 各模态权重，若不提供则使用当前权重
            node_id (int, optional): 节点ID，若不提供则使用默认节点
            
        Returns:
            torch.Tensor: 融合后的数据
        """
        # 如果未提供节点ID，默认使用最后一个节点（多模态节点）
        if node_id is None:
            node_id = self.num_nodes - 1
            
        # 如果未提供权重，使用指定节点的权重
        if weights is None:
            if node_id < self.num_nodes:
                weights = self.modality_weights[node_id]
            else:
                # 节点ID超出范围，使用默认权重
                weights = np.ones(self.num_modalities) / self.num_modalities
        
        # 确保权重和数据模态数量匹配
        if len(weights) != len(multimodal_data):
            weights = np.ones(len(multimodal_data)) / len(multimodal_data)
        
        # 标准化权重
        weights = weights / np.sum(weights)
        
        # 初始化融合结果
        fused_data = None
        
        # 按权重融合多模态数据
        for i, (modality, data) in enumerate(multimodal_data.items()):
            if fused_data is None:
                fused_data = weights[i] * data
            else:
                # 确保数据形状匹配
                if isinstance(data, torch.Tensor) and isinstance(fused_data, torch.Tensor):
                    if data.shape != fused_data.shape:
                        # 如果形状不一致，将数据重塑为相同维度
                        data = data.view(fused_data.shape)
                    fused_data = fused_data + weights[i] * data
        
        return fused_data
    
    def update_modality_weights(self, node_id, modality_metrics):
        """
        更新模态权重，基于图片中的w_i(t)公式
        
        Args:
            node_id (int): 节点ID
            modality_metrics (dict): 各模态性能指标，格式为 {modality: {'accuracy': float, 'uncertainty': float}}
        """
        if node_id >= self.num_nodes:
            return
        
        # 提取各模态的准确率和不确定性
        accuracies = []
        uncertainties = []
        
        for modality in self.modalities:
            if modality in modality_metrics:
                accuracies.append(modality_metrics[modality].get('accuracy', 0.5))
                uncertainties.append(modality_metrics[modality].get('uncertainty', 1.0))
            else:
                accuracies.append(0.5)  # 默认准确率
                uncertainties.append(1.0)  # 默认不确定性
        
        # 将列表转换为numpy数组
        accuracies = np.array(accuracies)
        uncertainties = np.array(uncertainties)
        
        # 计算模态权重：准确率高、不确定性低的模态获得更高权重
        # w_i(t) ∝ accuracy_i / uncertainty_i
        weights = accuracies / (uncertainties + 1e-6)  # 添加小量避免除零
        
        # 归一化权重
        weights = weights / (np.sum(weights) + 1e-6)
        
        # 更新该节点的模态权重
        self.modality_weights[node_id] = weights
        
        # 打印权重更新信息
        modality_weight_str = ", ".join([f"{m}: {w:.3f}" for m, w in zip(self.modalities, weights)])
        print(f"节点{node_id}的模态权重已更新: {modality_weight_str}")
    
    def calculate_prediction_uncertainty(self, prediction_probs):
        """
        计算预测不确定性，基于图片中的信息熵公式 H(y)
        
        Args:
            prediction_probs (torch.Tensor): 预测概率分布，形状为 [batch_size, num_classes]
            
        Returns:
            torch.Tensor: 不确定性值 (信息熵)
        """
        # 确保概率和为1
        prediction_probs = nn.functional.softmax(prediction_probs, dim=1)
        
        # 计算信息熵: H(y) = -∑ P(y|z(t)) log P(y|z(t))
        eps = 1e-10  # 添加小量避免log(0)
        entropy = -torch.sum(prediction_probs * torch.log(prediction_probs + eps), dim=1)
        
        # 返回平均熵作为整体不确定性
        return entropy.mean().item()
    
    def adaptive_reinforcement_learning(self, state, action, uncertainty, reward):
        """
        基于不确定性的自适应强化学习，用于优化模型参数
        
        Args:
            state (numpy.ndarray): 当前状态，包含性能指标
            action (int): 执行的动作
            uncertainty (float): 当前预测的不确定性
            reward (float): 获得的奖励
            
        Returns:
            float: 更新后的Q值
        """
        # 定义状态空间 S (如图片所述)
        # 包含模型参数、预测结果和不确定性
        
        # 定义动作空间 A
        # 包括调整模型结构和更新权重参数
        
        # 定义奖励函数 r(s,a)
        # 根据动作类型和不确定性变化给予奖励或惩罚
        
        # Q网络学习率，随不确定性调整
        alpha = 0.1 * (1.0 - min(1.0, uncertainty))
        
        # 简化的Q学习更新
        # Q(s,a) = Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        q_value = reward + 0.95 * max(0.5, 1.0 - uncertainty)
        
        return q_value
    
    def update_node_weights(self, node_metrics):
        """
        更新节点权重，用于联邦聚合
        
        Args:
            node_metrics (list): 节点性能指标列表
        """
        # 提取各节点的准确率、延迟和不确定性
        accuracies = []
        latencies = []
        uncertainties = []
        
        for i, metrics in enumerate(node_metrics):
            if metrics is None or i >= self.num_nodes:
                continue
                
            accuracies.append(metrics.get('train_acc', 0.5))
            latencies.append(metrics.get('latency', 100.0))
            uncertainties.append(metrics.get('uncertainty', 0.5))
        
        # 转换为numpy数组
        accuracies = np.array(accuracies)
        latencies = np.array(latencies)
        uncertainties = np.array(uncertainties)
        
        # 归一化延迟 (越低越好)
        max_latency = np.max(latencies) if np.max(latencies) > 0 else 1.0
        norm_latencies = 1.0 - (latencies / max_latency)
        
        # 计算综合评分: 准确率高、延迟低、不确定性低的节点获得更高权重
        scores = 0.6 * accuracies + 0.2 * norm_latencies + 0.2 * (1.0 - uncertainties)
        
        # 归一化得到权重
        weights = scores / np.sum(scores)
        
        # 更新节点权重
        self.node_weights[:len(weights)] = weights
        
        # 打印权重更新信息
        weight_str = ", ".join([f"节点{i}: {w:.3f}" for i, w in enumerate(weights)])
        print(f"联邦聚合节点权重已更新: {weight_str}")
        
        return self.node_weights.copy()
    
    def update(self, data):
        """
        更新态势感知数据
        
        Args:
            data (dict): 包含最新状态数据的字典，包括：
                - round: 当前轮次
                - node_accuracies: 各节点准确率
                - node_latencies: 各节点延迟
                - node_uncertainties: 各节点预测不确定性
                - modality_metrics: 各节点的模态性能指标
        """
        # 更新轮次
        if 'round' in data:
            epoch = data['round']
            if 'epochs' not in self.metrics_history:
                self.metrics_history['epochs'] = []
            
            if len(self.metrics_history['epochs']) <= epoch:
                self.metrics_history['epochs'].append(epoch)
        else:
            # 如果未提供轮次，使用当前历史记录长度
            epoch = len(self.metrics_history['epochs'])
            self.metrics_history['epochs'].append(epoch)
        
        # 更新节点准确率
        if 'node_accuracies' in data:
            for node_id, accuracy in data['node_accuracies'].items():
                node_id = int(node_id)
                if node_id < self.num_nodes:
                    while len(self.metrics_history['accuracy'][node_id]) <= epoch:
                        self.metrics_history['accuracy'][node_id].append(0.0)
                    self.metrics_history['accuracy'][node_id][epoch] = accuracy
        
        # 更新节点延迟
        if 'node_latencies' in data:
            for node_id, latency in data['node_latencies'].items():
                node_id = int(node_id)
                if node_id < self.num_nodes:
                    while len(self.metrics_history['latency'][node_id]) <= epoch:
                        self.metrics_history['latency'][node_id].append(0.0)
                    self.metrics_history['latency'][node_id][epoch] = latency
        
        # 更新节点不确定性
        if 'node_uncertainties' in data:
            for node_id, uncertainty in data['node_uncertainties'].items():
                node_id = int(node_id)
                if node_id < self.num_nodes:
                    while len(self.metrics_history['uncertainty'][node_id]) <= epoch:
                        self.metrics_history['uncertainty'][node_id].append(0.5)
                    self.metrics_history['uncertainty'][node_id][epoch] = uncertainty
        
        # 更新各节点的模态权重
        if 'modality_metrics' in data:
            for node_id, metrics in data['modality_metrics'].items():
                node_id = int(node_id)
                if node_id < self.num_nodes:
                    self.update_modality_weights(node_id, metrics)
        
        # 更新节点权重
        if 'node_metrics' in data:
            self.update_node_weights(data['node_metrics'])
    
    def generate_fusion_report(self):
        """
        生成模态融合报告
        
        Returns:
            dict: 包含融合报告的字典
        """
        report = {
            'modality_weights': {},
            'node_weights': self.node_weights.copy(),
            'performance_gain': {},
            'uncertainty_reduction': {}
        }
        
        # 收集各节点的模态权重
        for node_id in range(self.num_nodes):
            report['modality_weights'][node_id] = self.modality_weights[node_id].copy()
        
        # 计算多模态融合带来的性能提升
        # 根据历史数据计算单模态vs多模态的性能差距
        
        # 假设最后一个节点是多模态模型，其它节点是单模态模型
        multimodal_node_id = self.num_nodes - 1  # 最后一个节点ID
        
        # 检查是否有足够的节点和历史数据
        if (self.num_nodes > 1 and 
            all(len(self.metrics_history['accuracy'][i]) > 0 for i in range(self.num_nodes))):
            
            # 计算单模态节点的平均准确率（不包括最后一个节点）
            single_modal_acc = np.mean([
                self.metrics_history['accuracy'][i][-1] 
                for i in range(self.num_nodes - 1)  # 除去最后一个多模态节点
                if self.metrics_history['accuracy'][i]
            ])
            
            # 获取多模态节点的准确率
            multi_modal_acc = self.metrics_history['accuracy'][multimodal_node_id][-1] if self.metrics_history['accuracy'][multimodal_node_id] else 0
            
            # 计算性能提升
            performance_gain = multi_modal_acc - single_modal_acc
            report['performance_gain'] = performance_gain
        
        # 计算不确定性减少
        if (self.num_nodes > 1 and 
            all(len(self.metrics_history['uncertainty'][i]) > 0 for i in range(self.num_nodes))):
            
            # 计算单模态节点的平均不确定性
            single_modal_unc = np.mean([
                self.metrics_history['uncertainty'][i][-1] 
                for i in range(self.num_nodes - 1)  # 除去最后一个多模态节点
                if self.metrics_history['uncertainty'][i]
            ])
            
            # 获取多模态节点的不确定性
            multi_modal_unc = self.metrics_history['uncertainty'][multimodal_node_id][-1] if self.metrics_history['uncertainty'][multimodal_node_id] else 0
            
            # 计算不确定性减少
            uncertainty_reduction = single_modal_unc - multi_modal_unc
            report['uncertainty_reduction'] = uncertainty_reduction
        
        return report
    
    def visualize_fusion_performance(self):
        """
        可视化模态融合性能
        """
        if not self.visualize:
            return
            
        # 假设最后一个节点是多模态节点
        multimodal_node_id = self.num_nodes - 1
        
        # 创建图形
        plt.figure(figsize=(12, 10))
        
        # 1. 模态权重演变
        plt.subplot(2, 2, 1)
        
        # 为多模态节点绘制权重演变
        node_id = multimodal_node_id  # 多模态融合节点
        
        # 检查是否有足够的历史数据
        if node_id < len(self.modality_weights) and len(self.modalities) > 0:
            weights = self.modality_weights[node_id]
            
            # 绘制饼图
            plt.pie(weights, labels=self.modalities, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'节点{node_id}的模态权重分布', fontsize=14)
        
        # 2. 不确定性与准确率关系
        plt.subplot(2, 2, 2)
        
        # 收集所有节点的不确定性和准确率
        uncertainties = []
        accuracies = []
        
        for node_id in range(self.num_nodes):
            if (node_id < len(self.metrics_history['uncertainty']) and 
                node_id < len(self.metrics_history['accuracy']) and
                len(self.metrics_history['uncertainty'][node_id]) > 0 and
                len(self.metrics_history['accuracy'][node_id]) > 0):
                uncertainties.append(self.metrics_history['uncertainty'][node_id][-1])
                accuracies.append(self.metrics_history['accuracy'][node_id][-1])
        
        if uncertainties and accuracies:
            # 绘制散点图
            plt.scatter(uncertainties, accuracies, s=100, alpha=0.7, c=range(len(uncertainties)), cmap='viridis')
            
            # 添加节点标签
            for i, (x, y) in enumerate(zip(uncertainties, accuracies)):
                plt.annotate(f'节点{i}', (x, y), xytext=(5, 5), textcoords='offset points')
            
            # 拟合简单线性回归线 (替代scipy.stats.linregress)
            try:
                # 简单的线性回归实现
                if len(uncertainties) > 1:
                    # 计算均值
                    mean_x = sum(uncertainties) / len(uncertainties)
                    mean_y = sum(accuracies) / len(accuracies)
                    
                    # 计算斜率和截距
                    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(uncertainties, accuracies))
                    denominator = sum((x - mean_x) ** 2 for x in uncertainties)
                    
                    if denominator != 0:
                        slope = numerator / denominator
                        intercept = mean_y - slope * mean_x
                        
                        # 绘制趋势线
                        x_range = np.linspace(min(uncertainties), max(uncertainties), 100)
                        y_range = intercept + slope * x_range
                        plt.plot(x_range, y_range, 'r--')
            except Exception as e:
                print(f"绘制趋势线时出错: {e}")
            
            plt.xlabel('预测不确定性', fontsize=12)
            plt.ylabel('准确率', fontsize=12)
            plt.title('不确定性与准确率关系', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        # 3. 多模态vs单模态性能比较
        plt.subplot(2, 2, 3)
        
        # 计算平均性能
        single_modal_acc_history = []
        multi_modal_acc_history = []
        
        # 多模态节点是最后一个节点
        for epoch in range(len(self.metrics_history['epochs'])):
            # 计算单模态节点的平均准确率（不包括最后一个节点）
            single_accs = []
            for i in range(self.num_nodes - 1):  # 除去最后一个多模态节点
                if (i < len(self.metrics_history['accuracy']) and 
                    len(self.metrics_history['accuracy'][i]) > epoch):
                    single_accs.append(self.metrics_history['accuracy'][i][epoch])
            
            if single_accs:
                single_modal_acc_history.append(np.mean(single_accs))
            else:
                single_modal_acc_history.append(0)
            
            # 获取多模态节点的准确率
            if (multimodal_node_id < len(self.metrics_history['accuracy']) and 
                len(self.metrics_history['accuracy'][multimodal_node_id]) > epoch):
                multi_modal_acc_history.append(self.metrics_history['accuracy'][multimodal_node_id][epoch])
            else:
                multi_modal_acc_history.append(0)
        
        epochs = range(1, len(self.metrics_history['epochs']) + 1)
        if epochs:
            plt.plot(epochs, single_modal_acc_history, 'o-', label='单模态平均', color='blue')
            plt.plot(epochs, multi_modal_acc_history, 's-', label='多模态融合', color='red')
            plt.xlabel('训练轮次', fontsize=12)
            plt.ylabel('准确率', fontsize=12)
            plt.title('多模态vs单模态性能比较', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. 节点权重变化
        plt.subplot(2, 2, 4)
        
        # 绘制节点权重
        labels = [f'节点{i}' for i in range(self.num_nodes)]
        plt.bar(labels, self.node_weights, color='skyblue')
        plt.xlabel('边缘节点', fontsize=12)
        plt.ylabel('聚合权重', fontsize=12)
        plt.title('联邦学习节点权重分布', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 保存图表
        try:
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/fusion_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"融合性能可视化图表已保存到 {self.output_dir}/fusion_performance.png")
        except Exception as e:
            print(f"保存融合性能可视化图表时出错: {e}")
            plt.close()
    
    def get_fusion_config(self):
        """
        获取当前的融合配置
        
        Returns:
            dict: 融合配置
        """
        return {
            'modality_weights': {i: weights.tolist() for i, weights in self.modality_weights.items()},
            'node_weights': self.node_weights.tolist(),
            'modalities': self.modalities
        }
    
    def update_global_metrics(self, round_num, metrics):
        """
        更新全局模型指标
        
        Args:
            round_num (int): 训练轮次
            metrics (dict): 指标字典
        """
        # 确保epoch列表长度足够
        while len(self.metrics_history['epochs']) < round_num:
            self.metrics_history['epochs'].append(len(self.metrics_history['epochs']))
        
        # 更新全局准确率
        if 'test_acc' in metrics:
            while len(self.metrics_history['global_acc']) < round_num:
                self.metrics_history['global_acc'].append(0.0)
            self.metrics_history['global_acc'][round_num-1] = metrics['test_acc']
        
        # 更新全局损失
        if 'test_loss' in metrics:
            while len(self.metrics_history['global_loss']) < round_num:
                self.metrics_history['global_loss'].append(0.0)
            self.metrics_history['global_loss'][round_num-1] = metrics['test_loss']

    def set_chinese_font(self):
        """设置中文字体，用于可视化图表"""
        import matplotlib.font_manager as fm
        
        # 尝试设置中文字体
        try:
            # 在不同系统上尝试不同的中文字体
            font_names = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'WenQuanYi Micro Hei']
            found_font = False
            
            for font_name in font_names:
                # 检查字体是否存在
                font_path = ''
                for f in fm.findSystemFonts():
                    if font_name in f and ('.ttf' in f or '.ttc' in f):
                        font_path = f
                        break
                
                if font_path:
                    # 设置默认字体
                    plt.rcParams['font.family'] = ['sans-serif']
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    found_font = True
                    break
            
            if not found_font:
                # 如果找不到任何中文字体，打印警告
                print("警告: 未找到合适的中文字体，图表标题和标签可能显示为方块")
                return False
            
            return True
        
        except Exception as e:
            print(f"设置中文字体时出错: {e}")
            return False 