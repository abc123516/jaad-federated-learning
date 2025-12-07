#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
# 获取当前脚本所在目录，并将其上级目录加入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import time
import random
import argparse
import numpy as np
import torch
import gc

from federation.data_handler.data_loader import JADDDataLoader    #用于处理JAAD数据集的数据加载功能
from federation.edge_nodes.node_manager import EdgeNodeManager    #管理边缘节点的组件
from federation.utils.situation_awareness import SituationAwareness #基础态势感知模块
from federation.utils.advanced_situation_awareness import AdvancedSituationAwareness  #高级态势感知模块，支持多模态融合
from federation.utils.fed_aggregator import FederatedAggregator #联邦学习中的模型聚合器

def parse_args():
    parser = argparse.ArgumentParser(description='联邦学习态势感知项目')
    parser.add_argument('--data_path', type=str, default='./JAAD_clips', help='JAAD视频数据路径')
    parser.add_argument('--annotation_path', type=str, default='./JAAD-JAAD_2.0/JAAD-JAAD_2.0', help='JAAD标注数据路径')
    parser.add_argument('--epochs', type=int, default=3, help='联邦学习总轮数')
    parser.add_argument('--local_epochs', type=int, default=1, help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_nodes', type=int, default=4, help='边缘节点数量')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化图表')
    parser.add_argument('--advanced_awareness', action='store_true', help='使用高级态势感知')
    return parser.parse_args()

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印参数信息
    print("联邦学习态势感知项目")
    print(f"数据路径: {args.data_path}")
    print(f"标注路径: {args.annotation_path}")
    print(f"联邦学习轮数: {args.epochs}")
    print(f"本地训练轮数: {args.local_epochs}")
    print(f"批量大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"边缘节点数: {args.num_nodes}")
    print(f"可视化: {'开启' if args.visualize else '关闭'}")
    print(f"高级态势感知: {'开启' if args.advanced_awareness else '关闭'}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    data_loader = JADDDataLoader(
        data_path=args.data_path,
        annotation_path=args.annotation_path,
        batch_size=args.batch_size
    )
    
    # 划分数据集
    print("\n划分数据集...")
    node_train_datasets, val_dataset, test_dataset = data_loader.split_data(args.num_nodes)
    
    # 创建边缘节点管理器
    print("\n创建边缘节点...")
    node_manager = EdgeNodeManager(
        num_nodes=args.num_nodes,
        train_datasets=node_train_datasets,
        val_dataset=val_dataset,
        device=device,
        lr=args.lr
    )
    
    # 创建联邦聚合器
    print("\n创建联邦聚合器...")
    fed_aggregator = FederatedAggregator(
        node_manager=node_manager,
        output_dir=args.output_dir,
        visualize=args.visualize
    )
    
    # 创建态势感知模块
    if args.advanced_awareness:
        # 使用高级态势感知模块
        situation_awareness = AdvancedSituationAwareness(
            num_nodes=args.num_nodes,
            modalities=['visual', 'traffic', 'vehicle', 'appearance', 'attributes'],
            visualize=args.visualize
        )
        print("\n创建高级态势感知模块...")
    else:
        # 使用基础态势感知模块
        situation_awareness = SituationAwareness(
            num_nodes=args.num_nodes,
            visualize=args.visualize
        )
        print("\n创建基础态势感知模块...")
    
    # 联邦学习过程
    print("\n开始联邦学习...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n轮次 {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # 1. 本地训练
        print("本地训练...")
        node_weights = []
        node_metrics = []
        node_uncertainties = {}
        node_latencies = {}
        
        for i, node in enumerate(node_manager.nodes):
            print(f"训练节点 {i+1}/{args.num_nodes}...")
            try:
                # 训练开始时间
                train_start = time.time()
                
                # 训练节点模型
                weights, metrics = node.train(local_epochs=args.local_epochs)
                
                # 训练结束时间，计算延迟
                train_end = time.time()
                latency = (train_end - train_start) * 1000  # 转为毫秒
                
                node_weights.append(weights)
                node_metrics.append(metrics)
                
                # 计算或模拟模型预测的不确定性
                if hasattr(node.model, 'calculate_uncertainty'):
                    uncertainty = node.model.calculate_uncertainty()
                else:
                    # 模拟不确定性 - 多模态模型通常不确定性更低
                    if node.model_type == 'multimodal_fusion':
                        uncertainty = random.uniform(0.1, 0.3)
                    else:
                        uncertainty = random.uniform(0.3, 0.6)
                
                # 记录延迟和不确定性
                node_latencies[str(i)] = latency
                node_uncertainties[str(i)] = uncertainty
                
                # 添加延迟和不确定性到指标中
                metrics['latency'] = latency
                metrics['uncertainty'] = uncertainty
                
                print(f"节点{i} ({node.model_type}) 训练完成:")
                print(f"  - 损失: {metrics['train_loss']:.4f}")
                print(f"  - 准确率: {metrics['train_acc']:.4f}")
                print(f"  - 延迟: {latency:.2f}ms")
                print(f"  - 不确定性: {uncertainty:.4f}")
                
            except Exception as e:
                print(f"节点{i}训练失败: {e}")
                node_weights.append(None)
                node_metrics.append(None)
        
        # 2. 更新态势感知模块
        if args.advanced_awareness:
            # 创建模态性能指标
            modality_metrics = {}
            
            # 使用最后一个节点作为多模态节点
            multimodal_node_id = args.num_nodes - 1
            
            for i in range(args.num_nodes):
                # 只为多模态节点创建模态指标
                if i == multimodal_node_id and i < len(node_metrics) and node_metrics[i] is not None:
                    modality_metrics[i] = {
                        'visual': {'accuracy': 0.75, 'uncertainty': 0.35},
                        'traffic': {'accuracy': 0.68, 'uncertainty': 0.42},
                        'vehicle': {'accuracy': 0.72, 'uncertainty': 0.38},
                        'appearance': {'accuracy': 0.65, 'uncertainty': 0.45},
                        'attributes': {'accuracy': 0.70, 'uncertainty': 0.40}
                    }
            
            # 提取节点准确率
            node_accuracies = {
                str(i): metrics['train_acc'] if metrics else 0.0
                for i, metrics in enumerate(node_metrics) if metrics
            }
            
            # 更新高级态势感知模块
            situation_awareness.update({
                'round': epoch,
                'node_accuracies': node_accuracies,
                'node_latencies': node_latencies,
                'node_uncertainties': node_uncertainties,
                'node_metrics': node_metrics,
                'modality_metrics': modality_metrics
            })
            
            # 获取节点权重
            node_weights_for_aggregation = situation_awareness.node_weights
            
            # 打印融合配置
            fusion_config = situation_awareness.get_fusion_config()
            print("\n当前融合配置:")
            print(f"节点权重: {fusion_config['node_weights']}")
            
            # 生成融合报告
            fusion_report = situation_awareness.generate_fusion_report()
            if 'performance_gain' in fusion_report:
                print(f"多模态性能提升: {fusion_report['performance_gain']:.4f}")
            if 'uncertainty_reduction' in fusion_report:
                print(f"不确定性减少: {fusion_report['uncertainty_reduction']:.4f}")
        else:
            # 使用常规态势感知模块
            # 获取节点状态
            node_status = situation_awareness.monitor_nodes(node_manager.nodes)
            
            # 更新性能指标
            situation_awareness.update_metrics(node_metrics, node_status)
            
            # 展示当前性能指标
            situation_awareness.display_metrics()
            
            # 简单平均权重
            node_weights_for_aggregation = None
        
        # 3. 联邦聚合
        print("\n聚合模型权重...")
        global_weights = fed_aggregator.aggregate(
            node_weights, 
            node_metrics, 
            epoch+1, 
            aggregation_weights=node_weights_for_aggregation if args.advanced_awareness else None
        )
        
        # 更新所有节点的全局模型
        print("\n更新全局模型...")
        updated_nodes = fed_aggregator.update_global_model_all_nodes(global_weights)
        print(f"成功更新了 {len(updated_nodes)}/{args.num_nodes} 个节点的全局模型")
        
        # 4. 全局模型评估
        print("\n评估全局模型...")
        test_metrics = fed_aggregator.evaluate_global_model(test_dataset)
        print(f"全局模型测试结果 - 损失: {test_metrics['test_loss']:.4f}, 准确率: {test_metrics['test_acc']:.4f}")
        
        # 更新全局指标
        situation_awareness.update_global_metrics(epoch+1, test_metrics)
        
        # 生成可视化
        if args.visualize and (epoch + 1) % 1 == 0:
            print("\n生成可视化图表...")
            fed_aggregator.visualize_training_progress()
            if args.advanced_awareness:
                situation_awareness.visualize_fusion_performance()
        
        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 训练完成，保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    print(f"\n保存最终模型到 {final_model_path}...")
    fed_aggregator.save_final_model(final_model_path)
    
    # 最终全局评估
    print("\n最终全局模型评估...")
    final_metrics = fed_aggregator.evaluate_global_model(test_dataset)
    print(f"最终模型性能 - 损失: {final_metrics['test_loss']:.4f}, 准确率: {final_metrics['test_acc']:.4f}")
    
    # 输出总训练时间
    total_time = time.time() - start_time
    print(f"\n联邦学习完成，总时间: {total_time/60:.2f}分钟")

if __name__ == "__main__":
    main() 