#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import re
import copy

from federation.models.resnet_gru import create_model as create_resnet_gru
from federation.models.mobilenet_lstm import create_model as create_mobilenet_lstm
from federation.models.efficientnet_transformer import create_model as create_efficientnet_transformer
from federation.models.multimodal_fusion import create_model as create_multimodal_fusion
from federation.data_handler.data_loader import JADDDataLoader

class EdgeNode:
    """边缘计算节点类"""
    
    def __init__(self, node_id, model, dataset, device, lr=0.001):
        """
        初始化边缘节点
        
        Args:
            node_id (int): 节点ID
            model (nn.Module): 节点模型
            dataset: 节点训练数据集
            device: 训练设备
            lr (float): 学习率
        """
        self.node_id = node_id
        self.model = model
        self.dataset = dataset
        self.device = device
        
        # 创建数据加载器，减小批量大小和worker数量
        self.dataloader = DataLoader(
            dataset, 
            batch_size=4,  # 减小批量大小
            shuffle=True, 
            num_workers=0,  # 不使用多进程加载
            pin_memory=False  # 不使用pin_memory
        )
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 节点状态
        self.status = {
            'online': True,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'latency': 0.0,
            'accuracy': 0.0
        }
        
        # 确定模型类型
        self.model_type = self._get_model_type()
        
        print(f"节点{node_id}初始化完成，模型类型: {self.model_type}，数据集大小: {len(dataset)}")
    
    def _get_model_type(self):
        """
        确定当前节点使用的模型类型
        
        Returns:
            str: 模型类型标识符
        """
        # 获取模型类名
        class_name = self.model.__class__.__name__
        
        if 'ResNet' in class_name or 'GRU' in class_name:
            return 'resnet_gru'
        elif 'MobileNet' in class_name or 'LSTM' in class_name:
            return 'mobilenet_lstm'
        elif 'EfficientNet' in class_name or 'Transformer' in class_name:
            return 'efficientnet_transformer'
        elif 'MultimodalFusion' in class_name:
            return 'multimodal_fusion'
        else:
            # 如果无法直接通过类名识别，尝试通过权重键识别
            weights = self.model.get_weights()
            keys = list(weights.keys())
            
            if any('resnet_features' in k for k in keys):
                return 'resnet_gru'
            elif any('mobilenet_features' in k for k in keys):
                return 'mobilenet_lstm'
            elif any('features' in k and 'projection' in ' '.join(keys) for k in keys):
                return 'efficientnet_transformer'
            elif any('multimodal_fusion' in k for k in keys):
                return 'multimodal_fusion'
            else:
                # 如果仍然无法识别，返回一个基于类名的标识符
                return class_name.lower()
    
    def train(self, local_epochs=5):
        """
        本地训练模型
        
        Args:
            local_epochs (int): 本地训练轮数
        
        Returns:
            tuple: (模型权重, 训练指标)
        """
        try:
            # 导入tqdm进度条库
            from tqdm import tqdm
        except ImportError:
            # 如果没有安装tqdm，尝试安装
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
                from tqdm import tqdm
            except:
                print("警告: 无法安装tqdm库，将不显示进度条")
                tqdm = lambda x, **kwargs: x  # 定义一个空的tqdm函数
        
        # 确保模型处于训练模式
        self.model.train()
        
        # 训练指标
        metrics = {
            'train_loss': 0.0,
            'train_acc': 0.0,
            'node_id': self.node_id,
            'model_type': self.model_type  # 添加模型类型到指标中
        }
        
        # 开始训练
        total_samples = 0
        correct_predictions = 0
        
        try:
            for epoch in range(local_epochs):
                epoch_loss = 0.0
                
                # 创建进度条，显示本地训练进度
                dataloader_with_progress = tqdm(
                    self.dataloader,
                    desc=f"节点{self.node_id} ({self.model_type}) 轮次 {epoch+1}/{local_epochs}",
                    ncols=100,
                    leave=False  # 完成后不保留进度条
                )
                
                for batch_idx, batch_data in enumerate(dataloader_with_progress):
                    try:
                        # 检查是否是多模态模型
                        is_multimodal = self.model_type == 'multimodal_fusion'
                        
                        # 解析数据
                        if is_multimodal:
                            # 对于多模态模型，数据包含帧序列、多模态特征和标签
                            frames, multimodal_features, labels = batch_data
                            
                            # 将数据移到指定设备
                            frames = frames.to(self.device)
                            
                            # 将每个特征移到设备上
                            for feature_type in multimodal_features:
                                multimodal_features[feature_type] = multimodal_features[feature_type].to(self.device)
                            
                            labels = labels.to(self.device)
                            
                            # 清零梯度
                            self.optimizer.zero_grad()
                            
                            # 前向传播
                            outputs = self.model(frames, multimodal_features)
                        else:
                            # 对于非多模态模型，只有帧序列和标签
                            if len(batch_data) == 3:  # 如果数据加载器返回了多模态数据
                                frames, _, labels = batch_data
                            else:
                                frames, labels = batch_data
                            
                            # 将数据移到指定设备
                            frames = frames.to(self.device)
                            labels = labels.to(self.device)
                            
                            # 清零梯度
                            self.optimizer.zero_grad()
                            
                            # 前向传播
                            outputs = self.model(frames)
                        
                        # 计算损失
                        loss = self.criterion(outputs, labels)
                        
                        # 反向传播
                        loss.backward()
                        
                        # 更新权重
                        self.optimizer.step()
                        
                        # 累计损失
                        epoch_loss += loss.item()
                        
                        # 统计正确预测的样本数
                        _, predicted = torch.max(outputs, 1)
                        total_samples += labels.size(0)
                        correct_predictions += (predicted == labels).sum().item()
                        
                        # 更新进度条信息
                        current_accuracy = correct_predictions / total_samples
                        dataloader_with_progress.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'acc': f"{current_accuracy:.4f}"
                        })
                        
                        # 每处理10个批次释放一次缓存
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    except Exception as e:
                        print(f"节点{self.node_id}训练批次{batch_idx}时出错: {e}")
                        continue
                
                # 计算平均损失
                metrics['train_loss'] = epoch_loss / max(1, len(self.dataloader))
                
                # 显示每轮训练结束的结果
                current_accuracy = correct_predictions / max(1, total_samples)
                print(f"节点{self.node_id} 第{epoch+1}轮训练 - 损失: {metrics['train_loss']:.4f}, 准确率: {current_accuracy:.4f}")
        
        except Exception as e:
            print(f"节点{self.node_id}训练过程中出错: {e}")
        
        # 计算准确率
        metrics['train_acc'] = correct_predictions / max(1, total_samples)
        
        # 更新节点状态
        self.status['accuracy'] = metrics['train_acc']
        
        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 返回模型权重和训练指标
        return self.model.get_weights(), metrics
    
    def evaluate(self, dataloader):
        """
        评估模型
        
        Args:
            dataloader: 数据加载器
        
        Returns:
            dict: 评估指标
        """
        try:
            # 导入tqdm进度条库
            from tqdm import tqdm
        except ImportError:
            # 如果已经在train方法中定义了空的tqdm函数
            tqdm = lambda x, **kwargs: x
            
        # 确保模型处于评估模式
        self.model.eval()
        
        # 评估指标
        metrics = {
            'val_loss': 0.0,
            'val_acc': 0.0,
            'node_id': self.node_id,
            'model_type': self.model_type  # 添加模型类型到指标中
        }
        
        try:
            # 不计算梯度
            with torch.no_grad():
                total_loss = 0.0
                correct = 0
                total = 0
                
                # 创建评估进度条
                eval_progress = tqdm(
                    dataloader,
                    desc=f"节点{self.node_id} ({self.model_type}) 评估",
                    ncols=100,
                    leave=False
                )
                
                for batch_data in eval_progress:
                    # 检查是否是多模态模型
                    is_multimodal = self.model_type == 'multimodal_fusion'
                    
                    # 解析数据
                    if is_multimodal:
                        # 对于多模态模型，数据包含帧序列、多模态特征和标签
                        frames, multimodal_features, labels = batch_data
                        
                        # 将数据移到指定设备
                        frames = frames.to(self.device)
                        
                        # 将每个特征移到设备上
                        for feature_type in multimodal_features:
                            multimodal_features[feature_type] = multimodal_features[feature_type].to(self.device)
                        
                        labels = labels.to(self.device)
                        
                        # 前向传播
                        outputs = self.model(frames, multimodal_features)
                    else:
                        # 对于非多模态模型，只有帧序列和标签
                        if len(batch_data) == 3:  # 如果数据加载器返回了多模态数据
                            frames, _, labels = batch_data
                        else:
                            frames, labels = batch_data
                        
                        # 将数据移到指定设备
                        frames = frames.to(self.device)
                        labels = labels.to(self.device)
                        
                        # 前向传播
                        outputs = self.model(frames)
                    
                    # 计算损失
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # 更新进度条信息
                    current_accuracy = correct / total if total > 0 else 0
                    eval_progress.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{current_accuracy:.4f}"
                    })
                
                # 计算平均损失和准确率
                metrics['val_loss'] = total_loss / max(1, len(dataloader))
                metrics['val_acc'] = correct / max(1, total)
                
                # 显示评估结果
                print(f"节点{self.node_id} 评估结果 - 损失: {metrics['val_loss']:.4f}, 准确率: {metrics['val_acc']:.4f}")
        
        except Exception as e:
            print(f"节点{self.node_id}评估时出错: {e}")
        
        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return metrics
    
    def update_weights(self, weights):
        """
        更新模型权重
        
        Args:
            weights (dict): 模型权重字典
        
        Returns:
            bool: 更新是否成功
        """
        try:
            # 检查是否是模型类型分组的权重字典
            if self.model_type in weights and isinstance(weights[self.model_type], dict):
                # 仅更新与当前模型类型匹配的权重
                self.model.set_weights(weights[self.model_type])
                print(f"节点{self.node_id}更新全局模型权重 (键完全匹配)")
            else:
                # 尝试加载完整的权重字典
                # 获取当前模型的权重键
                model_keys = set(self.model.state_dict().keys())
                
                # 检查权重键是否匹配
                weight_keys = set(weights.keys())
                
                if model_keys == weight_keys:
                    # 键完全匹配
                    self.model.set_weights(weights)
                    print(f"节点{self.node_id}更新全局模型权重 (键完全匹配)")
                elif model_keys.issubset(weight_keys):
                    # 模型键是权重键的子集
                    matched_weights = {k: v for k, v in weights.items() if k in model_keys}
                    self.model.set_weights(matched_weights)
                    print(f"节点{self.node_id}更新全局模型权重 (部分键匹配)")
                else:
                    # 键不匹配，无法更新
                    print(f"节点{self.node_id}无法更新全局模型权重 (键不匹配)")
                    return False
            
            return True
        
        except Exception as e:
            print(f"节点{self.node_id}更新权重时出错: {e}")
            return False
    
    def update_status(self, status):
        """
        更新节点状态
        
        Args:
            status (dict): 状态字典
        """
        self.status.update(status)

class EdgeNodeManager:
    """边缘节点管理器"""
    
    def __init__(self, num_nodes, train_datasets, val_dataset, device, lr=0.001):
        """
        初始化边缘节点管理器
        
        Args:
            num_nodes (int): 边缘节点数量
            train_datasets (list): 训练数据集列表
            val_dataset: 验证数据集
            device: 训练设备
            lr (float): 学习率
        """
        self.num_nodes = num_nodes
        self.train_datasets = train_datasets
        self.val_dataset = val_dataset
        self.device = device
        self.lr = lr
        
        # 创建边缘节点
        self.nodes = self._create_nodes()
    
    def _create_nodes(self):
        """
        创建边缘计算节点
        
        Returns:
            list: 边缘节点列表
        """
        nodes = []
        
        # 为每个节点创建不同的模型
        for i in range(self.num_nodes):
            # 最后一个节点应该使用多模态融合模型，无论节点总数是多少
            if i == self.num_nodes - 1:
                # 最后一个节点：多模态融合模型
                model = create_multimodal_fusion(self.device)
                print(f"为节点{i}创建多模态融合模型")
            elif i == 0:
                # 节点1：ResNet-50 + GRU
                model = create_resnet_gru(self.device)
            elif i == 1:
                # 节点2：MobileNetV3 + LSTM
                model = create_mobilenet_lstm(self.device)
            elif i == 2:
                # 节点3：EfficientNet + Transformer
                model = create_efficientnet_transformer(self.device)
            else:
                # 额外节点：根据节点ID选择模型类型
                model_type = i % 3  # 使用3种单模态模型循环
                if model_type == 0:
                    model = create_resnet_gru(self.device)
                elif model_type == 1:
                    model = create_mobilenet_lstm(self.device)
                else:
                    model = create_efficientnet_transformer(self.device)
            
            # 创建节点
            node = EdgeNode(
                node_id=i,
                model=model,
                dataset=self.train_datasets[i],
                device=self.device,
                lr=self.lr
            )
            
            nodes.append(node)
        
        return nodes
    
    def get_model_type(self, node_id):
        """
        获取指定节点的模型类型
        
        Args:
            node_id (int): 节点ID
        
        Returns:
            str: 模型类型标识符
        """
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id].model_type
        return None
    
    def update_global_weights(self, global_weights):
        """
        更新所有节点的全局权重
        
        Args:
            global_weights: 全局权重（可能是单个权重字典或按模型类型分组的字典）
        
        Returns:
            list: 更新成功的节点ID列表
        """
        success_nodes = []
        
        # 检查是否接收到模型类型分组的权重字典
        is_grouped_weights = any(isinstance(v, dict) and all(isinstance(t, torch.Tensor) for t in v.values()) 
                                for v in global_weights.values())
        
        for node_id, node in enumerate(self.nodes):
            try:
                model_type = node.model_type
                
                # 如果是分组的权重，则选择与节点模型类型匹配的权重
                if is_grouped_weights and model_type in global_weights:
                    # 直接传递此类型的模型权重
                    success = node.update_weights(global_weights[model_type])
                    if success:
                        print(f"节点{node_id}更新了{model_type}类型的全局权重")
                        success_nodes.append(node_id)
                else:
                    # 向后兼容模式：尝试更新整个权重字典
                    success = node.update_weights(global_weights)
                    if success:
                        success_nodes.append(node_id)
            except Exception as e:
                print(f"更新节点{node_id}权重时出错: {e}")
        
        print(f"成功更新了{len(success_nodes)}/{len(self.nodes)}个节点的权重")
        return success_nodes
    
    def evaluate_global_model(self, dataset, batch_size=None):
        """
        评估全局模型
        
        Args:
            dataset: 评估数据集
            batch_size (int, optional): 批量大小
        
        Returns:
            dict: 评估指标
        """
        try:
            # 导入tqdm进度条库
            from tqdm import tqdm
        except ImportError:
            # 如果没有安装tqdm
            tqdm = lambda x, **kwargs: x
            
        # 如果未指定批量大小，使用默认值
        if batch_size is None:
            batch_size = 4  # 默认较小的批大小，减少内存使用
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 不使用多进程加载
            pin_memory=False  # 不使用pin_memory
        )
        
        # 评估指标
        metrics = {
            'test_loss': 0.0,
            'test_acc': 0.0
        }
        
        # 选择第一个节点进行评估
        if not self.nodes:
            print("错误: 没有可用的节点进行评估")
            return metrics
        
        eval_node = self.nodes[0]
        
        try:
            # 确保模型处于评估模式
            eval_node.model.eval()
            
            # 确保标准使用相同的设备
            eval_node.criterion = eval_node.criterion.to(eval_node.device)
            
            # 不计算梯度
            with torch.no_grad():
                total_loss = 0.0
                correct = 0
                total = 0
                
                # 创建评估进度条
                global_eval_progress = tqdm(
                    dataloader,
                    desc=f"全局模型评估",
                    ncols=100
                )
                
                for batch_data in global_eval_progress:
                    try:
                        # 解析数据
                        if len(batch_data) == 3:  # 如果数据加载器返回了多模态数据
                            frames, multimodal_features, labels = batch_data
                            
                            # 将数据移到指定设备
                            frames = frames.to(eval_node.device)
                            labels = labels.to(eval_node.device)  # 确保标签也在相同设备上
                            
                            # 将每个特征移到设备上（如果是多模态模型）
                            if eval_node.model_type == 'multimodal_fusion':
                                multimodal_features_device = {}
                                for feature_type, feature in multimodal_features.items():
                                    multimodal_features_device[feature_type] = feature.to(eval_node.device)
                                
                                outputs = eval_node.model(frames, multimodal_features_device)
                            else:
                                outputs = eval_node.model(frames)
                        else:
                            frames, labels = batch_data
                            
                            # 将数据移到指定设备
                            frames = frames.to(eval_node.device)
                            labels = labels.to(eval_node.device)
                            
                            # 前向传播
                            outputs = eval_node.model(frames)
                        
                        # 计算损失
                        loss = eval_node.criterion(outputs, labels)
                        total_loss += loss.item()
                        
                        # 计算准确率
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        # 更新进度条信息
                        current_accuracy = correct / total if total > 0 else 0
                        global_eval_progress.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'acc': f"{current_accuracy:.4f}"
                        })
                    except Exception as e:
                        print(f"评估批次时出错: {e}")
                        continue
                
                # 计算平均损失和准确率
                metrics['test_loss'] = total_loss / max(1, len(dataloader))
                metrics['test_acc'] = correct / max(1, total)
        
        except Exception as e:
            print(f"全局模型评估时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return metrics
    
    def transfer_task(self, from_node_id, to_node_id):
        """
        任务迁移：将一个节点的任务迁移到另一个节点
        
        Args:
            from_node_id (int): 源节点ID
            to_node_id (int): 目标节点ID
        
        Returns:
            bool: 迁移是否成功
        """
        if from_node_id >= len(self.nodes) or to_node_id >= len(self.nodes):
            print(f"错误: 无效的节点ID (from_node_id={from_node_id}, to_node_id={to_node_id})")
            return False
        
        # 检查两个节点的模型类型是否兼容
        source_type = self.get_model_type(from_node_id)
        target_type = self.get_model_type(to_node_id)
        
        if source_type != target_type:
            print(f"警告: 源节点({source_type})和目标节点({target_type})模型类型不同，迁移后可能存在兼容问题")
        
        # 获取源节点的数据集
        source_dataset = self.nodes[from_node_id].dataset
        
        # 创建一个新的数据集，包含两个节点的数据
        combined_dataset = torch.utils.data.ConcatDataset([
            self.nodes[to_node_id].dataset,
            source_dataset
        ])
        
        # 更新目标节点的数据集
        self.nodes[to_node_id].dataset = combined_dataset
        
        # 更新目标节点的数据加载器
        self.nodes[to_node_id].dataloader = DataLoader(
            combined_dataset, 
            batch_size=4,  # 减小批量大小
            shuffle=True, 
            num_workers=0,  # 不使用多进程
            pin_memory=False  # 不使用pin_memory
        )
        
        print(f"任务从节点{from_node_id}迁移到节点{to_node_id}, 目标节点数据量: {len(combined_dataset)}")
        
        return True
    
    def save_global_model(self, path):
        """
        保存全局模型
        
        Args:
            path (str): 保存路径
        """
        # 使用第一个节点的模型作为全局模型
        torch.save(self.nodes[0].model.state_dict(), path)
        print(f"全局模型保存到: {path}")
    
    def load_global_model(self, path):
        """
        加载全局模型
        
        Args:
            path (str): 模型路径
        
        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载模型权重
            weights = torch.load(path, map_location=self.device)
            
            # 更新所有节点的权重
            for node in self.nodes:
                node.update_weights(weights)
            
            print(f"从 {path} 成功加载全局模型")
            return True
        
        except Exception as e:
            print(f"加载全局模型时出错: {e}")
            return False 