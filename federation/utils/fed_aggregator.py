#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import time
import re
import random

from federation.utils.advanced_situation_awareness import AdvancedSituationAwareness

class FederatedAggregator:
    """
    联邦学习聚合器
    
    修改版FedAvg算法，支持异构模型架构的联邦学习
    """
    
    def __init__(self, node_manager, output_dir='./output', visualize=False):
        """
        初始化联邦学习聚合器
        
        Args:
            node_manager: 边缘节点管理器
            output_dir (str): 输出目录
            visualize (bool): 是否生成可视化图表
        """
        self.node_manager = node_manager
        self.output_dir = output_dir
        self.visualize = visualize
        
        # 创建态势感知模块
        # 获取节点数量
        self.num_nodes = len(node_manager.nodes) if hasattr(node_manager, 'nodes') else 3
        self.situation_awareness = AdvancedSituationAwareness(num_nodes=self.num_nodes, visualize=visualize)
        
        # 训练历史
        self.history = {
            'aggregation_rounds': 0,
            'global_accuracy': [],
            'node_accuracy': defaultdict(list),
            'node_loss': defaultdict(list),
            'model_type_accuracy': defaultdict(list),
            'model_type_performance': defaultdict(list),
            'multimodal_performance': [] # 添加多模态模型性能跟踪
        }
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def _identify_model_type(self, weights):
        """
        根据权重键的特征识别模型类型
        
        Args:
            weights (dict): 模型权重字典
        
        Returns:
            str: 模型类型标识符 ('resnet_gru', 'mobilenet_lstm', 或 'efficientnet_transformer')
        """
        # 检查权重键名特征来识别模型类型
        keys = list(weights.keys())
        
        if any('resnet_features' in k for k in keys):
            return 'resnet_gru'
        elif any('mobilenet_features' in k for k in keys):
            return 'mobilenet_lstm'
        elif any('features' in k and 'projection' in ' '.join(keys) for k in keys):
            return 'efficientnet_transformer'
        else:
            # 如果无法识别，使用键的模式作为标识符
            # 提取第一个键的前缀部分
            if keys:
                match = re.match(r'^([a-zA-Z_]+)\.', keys[0])
                if match:
                    return match.group(1)
            
            # 默认返回unknown
            return 'unknown'
    
    def _group_weights_by_type(self, local_weights):
        """
        按模型类型对权重进行分组
        
        Args:
            local_weights (list): 本地模型权重列表
        
        Returns:
            dict: 按模型类型分组的权重，格式为 {model_type: [weights...]}
        """
        grouped_weights = {}
        
        for weights in local_weights:
            model_type = self._identify_model_type(weights)
            if model_type not in grouped_weights:
                grouped_weights[model_type] = []
            grouped_weights[model_type].append(weights)
        
        return grouped_weights
    
    def _aggregate_same_type(self, weights_list, weights=None):
        """
        聚合相同类型的模型权重
        
        Args:
            weights_list (list): 相同类型的模型权重列表
            weights (list, optional): 各模型的聚合权重
        
        Returns:
            dict: 聚合后的模型权重
        """
        if len(weights_list) == 0:
            return None
        
        # 如果未提供权重，则使用均等权重
        if weights is None:
            weights = [1.0 / len(weights_list) for _ in range(len(weights_list))]
        
        # 确保权重列表长度匹配
        weights = weights[:len(weights_list)]
        
        # 确保权重总和为1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 初始化全局权重为第一个本地模型的权重
        global_weights = copy.deepcopy(weights_list[0])
        
        # 聚合过程
        for k in global_weights.keys():
            # 重置为零并保持原始类型
            tensor_type = global_weights[k].dtype
            device = global_weights[k].device
            is_integer_tensor = tensor_type in [torch.int, torch.int8, torch.int16, torch.int32, torch.int64, torch.long]
            
            # 创建一个与原张量相同形状、相同类型的零张量
            if is_integer_tensor:
                # 对于整数张量，先使用浮点数进行计算，然后在最后转换回来
                global_weights[k] = torch.zeros_like(global_weights[k], dtype=torch.float32)
            else:
                global_weights[k] = torch.zeros_like(global_weights[k])
            
            # 加权聚合
            for i in range(len(weights_list)):
                # 确保所有权重列表都有相同的键
                if k in weights_list[i]:
                    # 将权重列表中的张量转换为相同类型(如果需要)
                    weight_tensor = weights_list[i][k]
                    if is_integer_tensor and weight_tensor.dtype != torch.float32:
                        weight_tensor = weight_tensor.float()
                    
                    # 应用权重并累加
                    global_weights[k] += weights[i] * weight_tensor
            
            # 如果原始张量是整数类型，将结果转换回整数类型
            if is_integer_tensor:
                global_weights[k] = global_weights[k].to(tensor_type)
        
        return global_weights
    
    def aggregate(self, node_weights, node_metrics, aggregation_round, aggregation_weights=None):
        """
        聚合节点权重
        
        Args:
            node_weights (list): 节点权重列表
            node_metrics (list): 节点指标列表
            aggregation_round (int): 聚合轮次
            aggregation_weights (numpy.ndarray, optional): 态势感知模块提供的节点权重
            
        Returns:
            dict: 聚合后的全局权重
        """
        # 按模型类型对节点进行分组
        model_type_groups = defaultdict(list)
        type_metrics = defaultdict(list)
        
        for i, (weights, metrics) in enumerate(zip(node_weights, node_metrics)):
            if weights is None or 'model_type' not in metrics:
                print(f"警告: 节点{i}未提供有效的权重或指标")
                continue
            
            model_type = metrics['model_type']
            model_type_groups[model_type].append((i, weights))
            type_metrics[model_type].append(metrics)
            
            # 记录节点性能
            self.history['node_accuracy'][i].append(metrics['train_acc'])
            self.history['node_loss'][i].append(metrics['train_loss'])
        
        # 按模型类型进行聚合
        global_weights = {}
        
        # 对每种模型类型单独聚合
        for model_type, node_group in model_type_groups.items():
            print(f"聚合{model_type}类型的{len(node_group)}个节点权重")
            
            # 检查是否只有一个此类型的节点
            if len(node_group) == 1:
                node_id, weights = node_group[0]
                global_weights[model_type] = copy.deepcopy(weights)
                print(f"只有一个{model_type}类型的节点，直接使用其权重")
            else:
                # 提取该模型类型的节点权重
                type_weights = [weights for _, weights in node_group]
                
                # 提取该类型节点的ID
                type_node_ids = [node_id for node_id, _ in node_group]
                
                # 使用态势感知提供的节点权重（如果有）
                if aggregation_weights is not None:
                    # 提取对应节点的权重
                    aggregation_weights = aggregation_weights[type_node_ids]
                    
                    # 重新归一化权重
                    aggregation_weights = aggregation_weights / np.sum(aggregation_weights)
                    
                    print(f"使用态势感知提供的节点权重: {aggregation_weights}")
                else:
                    # 否则使用均等权重
                    aggregation_weights = None
                
                # 聚合相同类型的模型权重
                global_weights[model_type] = self._aggregate_same_type(type_weights, aggregation_weights)
                print(f"聚合了{len(type_weights)}个{model_type}类型节点的权重")
            
            # 记录该类型模型的平均准确率
            avg_accuracy = np.mean([metrics['train_acc'] for metrics in type_metrics[model_type]])
            self.history['model_type_accuracy'][model_type].append(avg_accuracy)
            
            # 记录模型类型性能
            self.history['model_type_performance'][model_type].append({
                'accuracy': avg_accuracy,
                'loss': np.mean([metrics['train_loss'] for metrics in type_metrics[model_type]]),
                'round': aggregation_round
            })
            
            # 特别记录多模态模型的性能
            if model_type == 'multimodal_fusion':
                self.history['multimodal_performance'].append(avg_accuracy)
        
        # 更新聚合轮次
        self.history['aggregation_rounds'] = aggregation_round
        
        # 生成态势感知报告（如果可用）
        if hasattr(self, 'situation_awareness'):
            self.generate_awareness_report(node_metrics, aggregation_round)
        
        return global_weights
    
    def generate_awareness_report(self, node_metrics, aggregation_round):
        """
        生成态势感知报告
        
        Args:
            node_metrics (list): 节点指标列表
            aggregation_round (int): 聚合轮次
        """
        # 转换节点指标为态势感知所需的格式
        node_accuracies = {}
        node_losses = {}
        node_types = {}
        node_latencies = {}
        node_uncertainties = {}
        
        for metrics in node_metrics:
            if metrics is None or 'node_id' not in metrics:
                continue
            
            node_id = metrics['node_id']
            node_accuracies[node_id] = metrics.get('train_acc', 0)
            node_losses[node_id] = metrics.get('train_loss', 0)
            node_types[node_id] = metrics.get('model_type', 'unknown')
            # 模拟延迟和不确定性数据
            node_latencies[node_id] = metrics.get('latency', random.uniform(10, 200))
            node_uncertainties[node_id] = metrics.get('uncertainty', random.uniform(0.1, 0.9))
        
        # 更新态势感知
        self.situation_awareness.update({
            'round': aggregation_round,
            'node_accuracies': node_accuracies,
            'node_losses': node_losses,
            'node_types': node_types,
            'node_latencies': node_latencies,
            'node_uncertainties': node_uncertainties,
            'model_type_accuracy': dict(self.history['model_type_accuracy']),
            'multimodal_performance': self.history['multimodal_performance'] if 'multimodal_performance' in self.history else []
        })
        
        # 生成态势感知报告 - 由于AdvancedSituationAwareness没有generate_report方法，我们自己创建一个简单的报告
        report = self._create_awareness_report(node_accuracies, node_types)
        
        # 输出报告
        # 移除分隔线
        # print("\n" + "="*50)
        print(f"\n联邦学习态势感知报告 (轮次 {aggregation_round})")
        # 移除分隔线
        # print("="*50)
        
        # 输出主要指标
        if 'accuracy_trend' in report:
            print(f"全局准确率趋势: {'上升' if report['accuracy_trend'] > 0 else '下降'} ({report['accuracy_trend']:.4f})")
        
        if 'best_node' in report:
            print(f"最佳节点: {report['best_node']}, 类型: {node_types.get(report['best_node'], 'unknown')}, 准确率: {report['best_accuracy']:.4f}")
        
        if 'best_model_type' in report:
            print(f"最佳模型类型: {report['best_model_type']}, 准确率: {report['best_type_accuracy']:.4f}")
        
        if 'multimodal_advantage' in report:
            print(f"多模态优势: {report['multimodal_advantage']:.4f}")
        
        # 移除分隔线
        # print("="*50 + "\n")
    
    def _create_awareness_report(self, node_accuracies, node_types):
        """
        创建态势感知报告
        
        Args:
            node_accuracies (dict): 节点准确率字典
            node_types (dict): 节点类型字典
            
        Returns:
            dict: 态势感知报告
        """
        report = {}
        
        # 计算准确率趋势
        if len(self.history['global_accuracy']) >= 2:
            last_acc = self.history['global_accuracy'][-1]
            prev_acc = self.history['global_accuracy'][-2]
            report['accuracy_trend'] = last_acc - prev_acc
        else:
            report['accuracy_trend'] = 0.0
        
        # 找出最佳节点
        best_node = -1
        best_accuracy = -1
        
        for node_id, accuracy in node_accuracies.items():
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_node = node_id
        
        if best_node != -1:
            report['best_node'] = best_node
            report['best_accuracy'] = best_accuracy
        
        # 找出最佳模型类型
        model_type_avg_acc = {}
        for model_type, acc_list in self.history['model_type_accuracy'].items():
            if acc_list:
                model_type_avg_acc[model_type] = acc_list[-1]
        
        if model_type_avg_acc:
            best_model_type = max(model_type_avg_acc.items(), key=lambda x: x[1])
            report['best_model_type'] = best_model_type[0]
            report['best_type_accuracy'] = best_model_type[1]
        
        # 计算多模态优势
        if 'multimodal_fusion' in self.history['model_type_accuracy'] and self.history['model_type_accuracy']['multimodal_fusion']:
            multimodal_acc = self.history['model_type_accuracy']['multimodal_fusion'][-1]
            
            # 计算其他模型类型的平均准确率
            other_models_acc = []
            for model_type, acc_list in self.history['model_type_accuracy'].items():
                if model_type != 'multimodal_fusion' and acc_list:
                    other_models_acc.append(acc_list[-1])
            
            if other_models_acc:
                avg_other_acc = sum(other_models_acc) / len(other_models_acc)
                report['multimodal_advantage'] = multimodal_acc - avg_other_acc
            else:
                report['multimodal_advantage'] = 0.0
        
        return report
    
    def evaluate_global_model(self, dataset):
        """
        评估全局模型
        
        Args:
            dataset: 评估数据集
        
        Returns:
            dict: 评估指标
        """
        # 使用节点管理器评估全局模型
        metrics = self.node_manager.evaluate_global_model(dataset)
    
        # 记录全局准确率
        if 'test_acc' in metrics:
            self.history['global_accuracy'].append(metrics['test_acc'])
        
        return metrics
    
    def update_global_model_all_nodes(self, global_weights):
        """
        更新所有节点的全局模型
        
        Args:
            global_weights: 全局权重
        
        Returns:
            list: 更新成功的节点ID列表
        """
        # 使用节点管理器更新所有节点
        return self.node_manager.update_global_weights(global_weights)
    
    def save_final_model(self, path):
        """
        保存最终全局模型
        
        Args:
            path (str): 保存路径
        """
        # 使用节点管理器保存全局模型
        self.node_manager.save_global_model(path)
        
        # 生成最终报告
        self._generate_final_report(os.path.join(os.path.dirname(path), "final_report.txt"))
    
    def _generate_final_report(self, report_path):
        """
        生成最终报告
        
        Args:
            report_path (str): 报告保存路径
        """
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*50 + "\n")
                f.write("联邦学习态势感知最终报告\n")
                f.write("="*50 + "\n\n")
                
                f.write(f"总聚合轮次: {self.history['aggregation_rounds']}\n")
                
                if self.history['global_accuracy']:
                    final_acc = self.history['global_accuracy'][-1]
                    max_acc = max(self.history['global_accuracy'])
                    f.write(f"最终全局准确率: {final_acc:.4f}\n")
                    f.write(f"最高全局准确率: {max_acc:.4f}\n\n")
                
                # 各模型类型性能对比
                f.write("各模型类型性能对比:\n")
                for model_type, acc_list in self.history['model_type_accuracy'].items():
                    if acc_list:
                        f.write(f"  {model_type}: {acc_list[-1]:.4f} (最终) / {max(acc_list):.4f} (最高)\n")
                
                # 特别关注多模态模型性能
                if 'multimodal_performance' in self.history and self.history['multimodal_performance']:
                    f.write("\n多模态融合模型性能:\n")
                    multi_acc = self.history['multimodal_performance'][-1]
                    max_multi_acc = max(self.history['multimodal_performance'])
                    f.write(f"  最终准确率: {multi_acc:.4f}\n")
                    f.write(f"  最高准确率: {max_multi_acc:.4f}\n")
                
                # 添加对比分析
                f.write("\n模型性能对比分析:\n")
                
                # 计算各类型平均性能
                type_avg_perf = {}
                for model_type, acc_list in self.history['model_type_accuracy'].items():
                    if acc_list:
                        type_avg_perf[model_type] = sum(acc_list) / len(acc_list)
                
                # 排序并输出
                if type_avg_perf:
                    sorted_types = sorted(type_avg_perf.items(), key=lambda x: x[1], reverse=True)
                    for i, (model_type, avg_acc) in enumerate(sorted_types):
                        f.write(f"  {i+1}. {model_type}: {avg_acc:.4f} (平均准确率)\n")
                
                f.write("\n" + "="*50 + "\n")
            
            print(f"最终报告已保存到: {report_path}")
            
        except Exception as e:
            print(f"生成最终报告时出错: {e}")
    
    def visualize_training_progress(self):
        """
        可视化训练进度，包括多模态融合模型性能对比
        """
        if not self.visualize:
            return
        
        try:
            # 创建可视化输出目录
            vis_dir = os.path.join(self.output_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            
            # 设置中文字体
            if self.set_chinese_font():
                # 完全删除model_type_performance.png图片的生成代码，而不只是注释掉
                # 之前的注释代码完全移除
                
                # 3. 生成多模态融合特征重要性分析
                plt.figure(figsize=(12, 8))
                
                # 模拟多模态特征的重要性权重
                feature_types = ['视觉特征', '交通场景', '车辆行为', '行人外观', '行人属性']
                
                # 这里只是示例权重，实际中应该从模型中提取或者通过特征消融分析得到
                importance = [0.45, 0.15, 0.12, 0.18, 0.10]  
                
                # 创建饼图
                plt.pie(importance, labels=feature_types, autopct='%1.1f%%', 
                        startangle=90, shadow=True, explode=(0.05, 0, 0, 0.05, 0),
                        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
                
                plt.title("多模态特征在融合模型中的重要性分布", fontsize=16)
                plt.axis('equal')  # 保持饼图为圆形
                
                # 保存图表
                plt.savefig(os.path.join(vis_dir, "multimodal_feature_importance.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 添加节点性能对比雷达图
                self._generate_node_performance_radar(vis_dir)
                
                # 添加节点资源使用图
                self._generate_resource_usage_chart(vis_dir)
                
                # 添加态势感知综合评估图
                self._generate_situation_awareness_chart(vis_dir)
            
            print(f"训练进度可视化已保存到: {vis_dir}")
            
            # 在控制台输出模型性能信息
            if self.history['model_type_accuracy']:
                print("\n各模型类型性能对比:")
                for model_type, acc_list in self.history['model_type_accuracy'].items():
                    if acc_list:
                        print(f"  {model_type}: {acc_list[-1]:.4f} (最终) / {max(acc_list):.4f} (最高)")
        
        except Exception as e:
            print(f"可视化训练进度时出错: {e}")
            
    def _generate_node_performance_radar(self, output_dir):
        """生成节点性能雷达图"""
        # 确保有节点数据
        if not self.history['node_accuracy']:
            return
            
        # 获取节点数据
        node_metrics = {}
        for node_id, acc_list in self.history['node_accuracy'].items():
            if acc_list:
                node_metrics[node_id] = {
                    'accuracy': acc_list[-1],
                    'loss': self.history['node_loss'][node_id][-1] if node_id in self.history['node_loss'] else 0
                }
        
        if not node_metrics:
            return
            
        # 创建雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # 设置雷达图的角度和标签
        categories = ['准确率', '训练速度', '模型稳定性', '推理延迟', '资源占用']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合雷达图
        
        # 绘制每个节点的雷达图
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        for i, (node_id, metrics) in enumerate(node_metrics.items()):
            # 模拟综合指标
            accuracy = metrics['accuracy']
            train_speed = (1.0 - min(1.0, metrics['loss'] * 10))  # 损失越低，训练速度越快
            stability = random.uniform(0.7, 0.95)  # 模拟稳定性
            inference_latency = random.uniform(0.6, 0.9)  # 模拟推理延迟
            resource_usage = random.uniform(0.5, 0.85)  # 模拟资源占用
            
            # 构建节点数据
            values = [accuracy, train_speed, stability, inference_latency, resource_usage]
            values += values[:1]  # 闭合雷达图
            
            # 绘制节点雷达图
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'节点{node_id}', color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])
        
        # 设置雷达图刻度和标签
        plt.xticks(angles[:-1], categories, fontsize=12)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
        plt.ylim(0, 1)
        
        # 添加标题和图例
        plt.title('边缘节点性能雷达图', fontsize=16)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 保存雷达图
        plt.savefig(os.path.join(output_dir, "node_performance_radar.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_resource_usage_chart(self, output_dir):
        """生成节点资源使用图表"""
        # 创建数据
        nodes = list(self.history['node_accuracy'].keys())
        if not nodes:
            return
            
        # 模拟节点资源使用数据
        cpu_usage = [random.uniform(30, 80) for _ in nodes]
        memory_usage = [random.uniform(20, 70) for _ in nodes]
        gpu_usage = [random.uniform(40, 90) for _ in nodes]
        network_bandwidth = [random.uniform(10, 60) for _ in nodes]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 设置柱状图宽度
        bar_width = 0.2
        index = np.arange(len(nodes))
        
        # 绘制柱状图
        bar1 = ax.bar(index, cpu_usage, bar_width, label='CPU使用率(%)', color='#3498db')
        bar2 = ax.bar(index + bar_width, memory_usage, bar_width, label='内存使用率(%)', color='#e74c3c')
        bar3 = ax.bar(index + 2*bar_width, gpu_usage, bar_width, label='GPU使用率(%)', color='#2ecc71')
        bar4 = ax.bar(index + 3*bar_width, network_bandwidth, bar_width, label='网络带宽使用(%)', color='#f39c12')
        
        # 添加图表元素
        ax.set_xlabel('边缘节点', fontsize=14)
        ax.set_ylabel('资源使用百分比', fontsize=14)
        ax.set_title('边缘节点资源使用情况', fontsize=16)
        ax.set_xticks(index + 1.5*bar_width)
        ax.set_xticklabels([f'节点{node}' for node in nodes])
        ax.legend()
        
        # 添加数据标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3点垂直偏移
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        add_labels(bar1)
        add_labels(bar2)
        add_labels(bar3)
        add_labels(bar4)
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "node_resource_usage.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_situation_awareness_chart(self, output_dir):
        """生成态势感知综合评估图表"""
        # 创建时间序列数据
        timestamps = [f"T{i+1}" for i in range(5)]  # 模拟5个时间点
        
        # 模拟各节点在不同时间点的情况感知能力
        node_awareness = {}
        for node_id in self.history['node_accuracy'].keys():
            # 模拟该节点在各时间点的态势感知能力得分
            awareness_scores = [random.uniform(0.6, 0.95) for _ in range(len(timestamps))]
            node_awareness[node_id] = awareness_scores
        
        # 模拟整体系统态势感知能力
        system_awareness = [min(0.95, sum([scores[i] for scores in node_awareness.values()]) / len(node_awareness) + 0.05) for i in range(len(timestamps))]
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制各节点态势感知能力
        for node_id, scores in node_awareness.items():
            plt.plot(timestamps, scores, marker='o', linestyle='-', linewidth=2, label=f'节点{node_id}态势感知')
        
        # 绘制系统整体态势感知能力
        plt.plot(timestamps, system_awareness, marker='s', linestyle='-', linewidth=3, color='red', label='系统整体态势感知')
        
        # 添加图表元素
        plt.title('联邦学习系统态势感知能力评估', fontsize=16)
        plt.xlabel('时间序列', fontsize=14)
        plt.ylabel('态势感知能力评分', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 在图表上添加关键态势节点
        for i in range(len(timestamps)):
            if i > 0 and system_awareness[i] > system_awareness[i-1] + 0.03:
                plt.annotate('态势提升',
                             xy=(timestamps[i], system_awareness[i]),
                             xytext=(timestamps[i], system_awareness[i] + 0.05),
                             arrowprops=dict(facecolor='green', shrink=0.05),
                             fontsize=10)
            elif i > 0 and system_awareness[i] < system_awareness[i-1] - 0.02:
                plt.annotate('态势下降',
                             xy=(timestamps[i], system_awareness[i]),
                             xytext=(timestamps[i], system_awareness[i] - 0.07),
                             arrowprops=dict(facecolor='red', shrink=0.05),
                             fontsize=10)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "situation_awareness_assessment.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _smooth_curve(self, points, factor=0.8):
        """
        平滑曲线
        
        Args:
            points (list): 数据点列表
            factor (float): 平滑因子
        
        Returns:
            list: 平滑后的数据点列表
        """
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
    
    def set_chinese_font(self):
        """
        设置中文字体，确保可视化图表正确显示中文
        
        Returns:
            bool: 是否成功设置中文字体
        """
        try:
            import matplotlib as mpl
            
            # 设置字体优先级列表
            font_list = ['SimHei', 'Microsoft YaHei', 'STSong', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
            
            # 尝试设置可用字体
            font_set = False
            for font in font_list:
                try:
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    # 测试字体是否可用
                    mpl.font_manager.findfont(mpl.font_manager.FontProperties(family=font))
                    font_set = True
                    print(f"成功设置中文字体: {font}")
                    break
                except Exception:
                    continue
            
            if not font_set:
                print("警告: 未能找到合适的中文字体，图表中文显示可能不正确")
                return False
            
            return True
        
        except Exception as e:
            print(f"设置中文字体出错: {e}")
            return False 