#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import random
import numpy as np
import psutil
import threading
import matplotlib.pyplot as plt
from collections import defaultdict
import os

class SituationAwareness:
    """态势感知模块，用于监控和评估边缘节点的性能指标"""
    
    def __init__(self, num_nodes, visualize=False):
        """
        初始化态势感知模块
        
        Args:
            num_nodes (int): 边缘节点数量
            visualize (bool): 是否生成可视化图表
        """
        self.num_nodes = num_nodes
        self.visualize = visualize
        
        # 初始化节点状态
        self.node_status = [True] * num_nodes
        
        # 存储性能指标历史数据
        self.metrics_history = {
            'accuracy': [[] for _ in range(num_nodes)],
            'cpu_usage': [[] for _ in range(num_nodes)],
            'memory_usage': [[] for _ in range(num_nodes)],
            'latency': [[] for _ in range(num_nodes)],
            'timestamps': []
        }
        
        # 初始化性能权重
        self.performance_weights = {
            'accuracy': 0.6,     # 准确率权重
            'cpu_usage': 0.15,   # CPU使用率权重(低使用率更好)
            'memory_usage': 0.1, # 内存使用率权重(低使用率更好)
            'latency': 0.15      # 延迟权重(低延迟更好)
        }
        
        # 初始化节点综合性能得分
        self.node_scores = [0.0] * num_nodes
        
        print(f"态势感知模块初始化完成，监控{num_nodes}个边缘节点")
        print(f"可视化模式: {'开启' if visualize else '关闭'}")
    
    def monitor_nodes(self, nodes):
        """
        监控边缘节点状态
        
        Args:
            nodes (list): 边缘节点列表
        
        Returns:
            list: 节点状态列表 (True表示在线，False表示离线)
        """
        # 当前时间
        current_time = time.time()
        self.metrics_history['timestamps'].append(current_time)
        
        # 模拟监控边缘节点
        for i in range(self.num_nodes):
            if not self.node_status[i]:
                # 节点离线，跳过监控
                continue
            
            # 模拟获取节点性能指标
            cpu_usage = random.uniform(10, 90)  # 模拟CPU使用率 (10%-90%)
            memory_usage = random.uniform(20, 80)  # 模拟内存使用率 (20%-80%)
            latency = random.uniform(10, 200)  # 模拟延迟 (10ms-200ms)
            
            # 存储性能指标
            self.metrics_history['cpu_usage'][i].append(cpu_usage)
            self.metrics_history['memory_usage'][i].append(memory_usage)
            self.metrics_history['latency'][i].append(latency)
            
            # 更新节点状态
            if i < len(nodes):
                nodes[i].update_status({
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'latency': latency
                })
        
        # 返回节点状态
        return self.node_status.copy()
    
    def update_metrics(self, local_metrics, node_status):
        """
        更新节点性能指标
        
        Args:
            local_metrics (list): 本地训练指标列表
            node_status (list): 节点状态列表
        """
        # 更新节点状态
        self.node_status = node_status.copy()
        
        # 更新准确率指标
        for metrics in local_metrics:
            node_id = metrics.get('node_id', 0)
            if node_id < self.num_nodes and self.node_status[node_id]:
                accuracy = metrics.get('train_acc', 0.0)
                self.metrics_history['accuracy'][node_id].append(accuracy)
        
        # 计算节点综合性能得分
        self._calculate_node_scores()
    
    def _calculate_node_scores(self):
        """计算节点综合性能得分"""
        for i in range(self.num_nodes):
            if not self.node_status[i]:
                # 节点离线，得分为0
                self.node_scores[i] = 0.0
                continue
            
            # 获取最新指标
            if self.metrics_history['accuracy'][i]:
                accuracy = self.metrics_history['accuracy'][i][-1]
            else:
                accuracy = 0.0
            
            if self.metrics_history['cpu_usage'][i]:
                cpu_usage = self.metrics_history['cpu_usage'][i][-1]
                # 转换为得分 (CPU使用率越低越好)
                cpu_score = 1.0 - (cpu_usage / 100.0)
            else:
                cpu_score = 0.0
            
            if self.metrics_history['memory_usage'][i]:
                memory_usage = self.metrics_history['memory_usage'][i][-1]
                # 转换为得分 (内存使用率越低越好)
                memory_score = 1.0 - (memory_usage / 100.0)
            else:
                memory_score = 0.0
            
            if self.metrics_history['latency'][i]:
                latency = self.metrics_history['latency'][i][-1]
                # 转换为得分 (延迟越低越好，假设最大延迟为500ms)
                latency_score = 1.0 - min(1.0, latency / 500.0)
            else:
                latency_score = 0.0
            
            # 计算加权得分
            score = (
                self.performance_weights['accuracy'] * accuracy +
                self.performance_weights['cpu_usage'] * cpu_score +
                self.performance_weights['memory_usage'] * memory_score +
                self.performance_weights['latency'] * latency_score
            )
            
            self.node_scores[i] = score
    
    def display_metrics(self):
        """显示节点性能指标"""
        print("\n节点性能指标:")
        print("节点ID\t状态\t  准确率\t  CPU使用率\t内存使用率\t延迟(ms)\t综合得分")
        print("-" * 50)
        
        for i in range(self.num_nodes):
            status = "在线" if self.node_status[i] else "离线"
            
            accuracy = self.metrics_history['accuracy'][i][-1] if self.metrics_history['accuracy'][i] else 0.0
            cpu_usage = self.metrics_history['cpu_usage'][i][-1] if self.metrics_history['cpu_usage'][i] else 0.0
            memory_usage = self.metrics_history['memory_usage'][i][-1] if self.metrics_history['memory_usage'][i] else 0.0
            latency = self.metrics_history['latency'][i][-1] if self.metrics_history['latency'][i] else 0.0
            
            print(f"{i}\t{status}\t  {accuracy:.4f}\t  {cpu_usage:.2f}%\t{memory_usage:.2f}%\t{latency:.2f}\t{self.node_scores[i]:.4f}")
    
    def get_node_scores(self):
        """获取节点综合性能得分"""
        return self.node_scores.copy()
    
    def handle_node_failure(self, failed_node_id, node_manager):
        """
        处理节点故障
        
        Args:
            failed_node_id (int): 故障节点ID
            node_manager: 节点管理器
        """
        # 标记节点为离线状态
        self.node_status[failed_node_id] = False
        
        # 寻找性能最好的在线节点
        best_node_id = -1
        best_score = -1
        
        for i in range(self.num_nodes):
            if i != failed_node_id and self.node_status[i] and self.node_scores[i] > best_score:
                best_node_id = i
                best_score = self.node_scores[i]
        
        if best_node_id >= 0:
            print(f"找到性能最好的在线节点: 节点{best_node_id}，得分: {best_score:.4f}")
            
            # 执行任务迁移
            success = node_manager.transfer_task(failed_node_id, best_node_id)
            
            if success:
                print(f"成功将任务从节点{failed_node_id}迁移到节点{best_node_id}")
            else:
                print(f"任务迁移失败")
        else:
            print("无法找到可用的节点进行任务迁移")
    
    def plot_metrics(self, save_path=None):
        """
        绘制性能指标图表
        
        Args:
            save_path (str, optional): 图表保存路径
        """
        # 如果可视化模式关闭，直接返回
        if not self.visualize:
            print("可视化模式已关闭，跳过绘制图表")
            return
            
        # 准备时间轴数据
        timestamps = range(len(self.metrics_history['timestamps']))
        
        # 创建画布
        fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
        
        # 绘制准确率
        for i in range(self.num_nodes):
            # 确保数据长度一致
            data_len = min(len(timestamps), len(self.metrics_history['accuracy'][i]))
            if data_len > 0:
                axs[0].plot(timestamps[:data_len], self.metrics_history['accuracy'][i][:data_len], 
                          label=f'节点{i}')
        axs[0].set_ylabel('准确率')
        axs[0].set_title('边缘节点准确率变化')
        axs[0].legend()
        axs[0].grid(True)
        
        # 绘制CPU使用率
        for i in range(self.num_nodes):
            data_len = min(len(timestamps), len(self.metrics_history['cpu_usage'][i]))
            if data_len > 0:
                axs[1].plot(timestamps[:data_len], self.metrics_history['cpu_usage'][i][:data_len], 
                          label=f'节点{i}')
        axs[1].set_ylabel('CPU使用率 (%)')
        axs[1].set_title('边缘节点CPU使用率变化')
        axs[1].legend()
        axs[1].grid(True)
        
        # 绘制内存使用率
        for i in range(self.num_nodes):
            data_len = min(len(timestamps), len(self.metrics_history['memory_usage'][i]))
            if data_len > 0:
                axs[2].plot(timestamps[:data_len], self.metrics_history['memory_usage'][i][:data_len], 
                          label=f'节点{i}')
        axs[2].set_ylabel('内存使用率 (%)')
        axs[2].set_title('边缘节点内存使用率变化')
        axs[2].legend()
        axs[2].grid(True)
        
        # 绘制延迟
        for i in range(self.num_nodes):
            data_len = min(len(timestamps), len(self.metrics_history['latency'][i]))
            if data_len > 0:
                axs[3].plot(timestamps[:data_len], self.metrics_history['latency'][i][:data_len], 
                          label=f'节点{i}')
        axs[3].set_ylabel('延迟 (ms)')
        axs[3].set_xlabel('时间')
        axs[3].set_title('边缘节点延迟变化')
        axs[3].legend()
        axs[3].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path)
        
        # 显示图表
        plt.close()
    
    def update_global_metrics(self, round_num, metrics):
        """
        更新全局模型性能指标
        
        Args:
            round_num (int): 当前轮次
            metrics (dict): 全局模型性能指标
        """
        # 这里可以添加存储全局模型性能指标的逻辑
        # 例如：准确率、损失等
        
        if not hasattr(self, 'global_metrics_history'):
            self.global_metrics_history = {
                'round': [],
                'accuracy': [],
                'loss': []
            }
        
        # 更新全局指标历史
        self.global_metrics_history['round'].append(round_num)
        self.global_metrics_history['accuracy'].append(metrics.get('test_acc', 0.0))
        self.global_metrics_history['loss'].append(metrics.get('test_loss', 0.0))
        
        # 如果开启可视化，绘制全局性能指标图表
        if self.visualize:
            self._plot_global_metrics()
        
        print(f"已更新全局模型第{round_num}轮性能指标")
    
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
    
    def _plot_global_metrics(self):
        """绘制全局模型性能指标图表"""
        if not hasattr(self, 'global_metrics_history') or not self.visualize:
            return
        
        # 设置中文字体
        self.set_chinese_font()
        
        # 以下代码已被注释，不再生成global_metrics图表
        # # 创建画布
        # plt.figure(figsize=(10, 6))
        # 
        # # 绘制准确率
        # plt.subplot(2, 1, 1)
        # plt.plot(
        #     self.global_metrics_history['round'], 
        #     self.global_metrics_history['accuracy'],
        #     marker='o'
        # )
        # plt.ylabel('准确率')
        # plt.title('全局模型准确率变化')
        # plt.grid(True)
        # 
        # # 绘制损失
        # plt.subplot(2, 1, 2)
        # plt.plot(
        #     self.global_metrics_history['round'], 
        #     self.global_metrics_history['loss'],
        #     marker='x'
        # )
        # plt.ylabel('损失')
        # plt.xlabel('轮次')
        # plt.title('全局模型损失变化')
        # plt.grid(True)
        # 
        # # 调整布局
        # plt.tight_layout()
        # 
        # # 创建输出目录（如果不存在）
        # os.makedirs('./output/visualization', exist_ok=True)
        # 
        # # 保存图表
        # save_path = f'./output/visualization/global_metrics_{len(self.global_metrics_history["round"])}.png'
        # plt.savefig(save_path)
        # 
        # # 关闭图表
        # plt.close()
        
        # 仅输出性能指标信息到控制台
        print(f"全局模型性能 - 准确率: {self.global_metrics_history['accuracy'][-1]:.4f}, 损失: {self.global_metrics_history['loss'][-1]:.4f}") 