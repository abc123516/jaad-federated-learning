#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from advanced_situation_awareness import AdvancedSituationAwareness

def simulate_prediction(model_type):
    """模拟不同模型的预测结果"""
    if model_type == 'multimodal':
        # 多模态模型预测更准确、更确定
        probs = torch.tensor([[0.05, 0.95], [0.1, 0.9], [0.08, 0.92]])
    else:
        # 单模态模型预测较不确定
        probs = torch.tensor([[0.25, 0.75], [0.3, 0.7], [0.4, 0.6]])
    
    return probs

def simulate_modality_metrics():
    """模拟不同模态的性能指标"""
    # 四种不同的模态性能
    metrics = {
        'visual': {'accuracy': 0.75, 'uncertainty': 0.35},
        'traffic': {'accuracy': 0.68, 'uncertainty': 0.42},
        'vehicle': {'accuracy': 0.72, 'uncertainty': 0.38},
        'appearance': {'accuracy': 0.65, 'uncertainty': 0.45},
        'attributes': {'accuracy': 0.70, 'uncertainty': 0.40}
    }
    return metrics

def main():
    """测试高级态势感知模块"""
    print("测试高级态势感知模块...")
    
    # 创建高级态势感知模块
    awareness = AdvancedSituationAwareness(
        num_nodes=4,
        modalities=['visual', 'traffic', 'vehicle', 'appearance', 'attributes'],
        visualize=True
    )
    
    # 模拟3轮训练
    for epoch in range(3):
        print(f"\n轮次 {epoch+1}:")
        
        # 为每个节点模拟性能指标
        node_accuracies = {}
        node_latencies = {}
        node_uncertainties = {}
        node_metrics = []
        modality_metrics = {}
        
        for node_id in range(4):
            # 模拟节点准确率 (多模态节点表现更好)
            if node_id == 3:  # 多模态节点
                accuracy = 0.85 + 0.05 * epoch  # 提高的准确率
                latency = 80 - 5 * epoch  # 降低的延迟
                
                # 模拟多模态预测
                prediction = simulate_prediction('multimodal')
                uncertainty = awareness.calculate_prediction_uncertainty(prediction)
                
                # 模拟多模态特征指标
                modality_metrics[node_id] = simulate_modality_metrics()
            else:  # 单模态节点
                accuracy = 0.70 + 0.03 * epoch
                latency = 60 - 3 * epoch
                
                # 模拟单模态预测
                prediction = simulate_prediction('single')
                uncertainty = awareness.calculate_prediction_uncertainty(prediction)
            
            # 记录指标
            node_accuracies[str(node_id)] = accuracy
            node_latencies[str(node_id)] = latency
            node_uncertainties[str(node_id)] = uncertainty
            
            # 添加到节点指标列表中
            node_metrics.append({
                'node_id': node_id,
                'train_acc': accuracy,
                'latency': latency,
                'uncertainty': uncertainty
            })
            
            print(f"节点{node_id} - 准确率: {accuracy:.4f}, 延迟: {latency:.2f}ms, 不确定性: {uncertainty:.4f}")
        
        # 更新态势感知模块
        awareness.update({
            'round': epoch,
            'node_accuracies': node_accuracies,
            'node_latencies': node_latencies,
            'node_uncertainties': node_uncertainties,
            'node_metrics': node_metrics,
            'modality_metrics': modality_metrics
        })
        
        # 获取融合配置
        fusion_config = awareness.get_fusion_config()
        print("\n当前融合配置:")
        print(f"节点权重: {fusion_config['node_weights']}")
        
        # 生成融合报告
        fusion_report = awareness.generate_fusion_report()
        print("\n融合性能报告:")
        if 'performance_gain' in fusion_report:
            print(f"多模态性能提升: {fusion_report['performance_gain']:.4f}")
        if 'uncertainty_reduction' in fusion_report:
            print(f"不确定性减少: {fusion_report['uncertainty_reduction']:.4f}")
    
    # 生成可视化图表
    awareness.visualize_fusion_performance()
    print("\n测试完成！")

if __name__ == "__main__":
    main() 