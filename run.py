#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
联邦学习态势感知项目主运行脚本
"""

import os
import sys
import argparse
import gc
import torch

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习态势感知项目')
    parser.add_argument('--data_path', type=str, default='./JAAD_clips', help='JAAD视频数据路径')
    parser.add_argument('--annotation_path', type=str, default='./JAAD-JAAD_2.0/JAAD-JAAD_2.0', help='JAAD标注数据路径')
    parser.add_argument('--epochs', type=int, default=3, help='联邦学习总轮数')
    parser.add_argument('--local_epochs', type=int, default=1, help='本地训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_nodes', type=int, default=3, help='边缘节点数量')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--memory_efficient', action='store_true', help='启用内存优化模式')
    parser.add_argument('--max_gpu_memory', type=float, default=0.6, help='最大GPU内存使用比例（0-1）')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化图表')
    parser.add_argument('--advanced_awareness', action='store_true', help='使用高级态势感知')
    return parser.parse_args()

def print_memory_stats():
    """打印内存使用统计信息"""
    # 注释掉CUDA内存相关信息的打印
    # if torch.cuda.is_available():
    #     print(f"CUDA内存分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    #     print(f"CUDA内存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # 强制垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def setup_memory_optimization():
    """设置内存优化"""
    # 设置PyTorch内存分配器的选项
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        # 限制缓存增长
        torch.cuda.set_per_process_memory_fraction(0.6)
    
    # 设置更积极的垃圾回收
    gc.enable()
    if hasattr(gc, 'set_threshold'):
        # 降低垃圾回收阈值，使其更频繁地触发
        threshold = gc.get_threshold()
        gc.set_threshold(threshold[0]//2, threshold[1]//2, threshold[2]//2)
    
    # 设置OpenCV环境变量以减少内存使用
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    # 禁用OpenCV的并行化
    os.environ['OPENCV_FOR_THREADS_NUM'] = '1'
    
    # 减少NumPy线程
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # 设置CUDA环境变量
    if torch.cuda.is_available():
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 应用内存优化配置
    if args.memory_efficient:
        setup_memory_optimization()
    
    # 限制CUDA内存使用
    if torch.cuda.is_available():
        # 限制GPU内存使用比例
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory, device)
    
    # 移除分隔线
    # print("=" * 50)
    print("联邦学习态势感知项目初始化")
    print(f"内存优化模式: {'启用' if args.memory_efficient else '禁用'}")
    print(f"GPU内存使用限制: {args.max_gpu_memory * 100:.0f}%")
    print(f"可视化图表: {'启用' if args.visualize else '禁用'}")
    # 移除分隔线
    # print("=" * 50)
    
    # 打印初始内存状态
    print_memory_stats()
    
    try:
        # 导入主模块
        from federation.main import main as fed_main
        
        # 准备参数传递到联邦学习主程序
        fed_args = [
            f"--data_path={args.data_path}",
            f"--annotation_path={args.annotation_path}",
            f"--epochs={args.epochs}",
            f"--local_epochs={args.local_epochs}",
            f"--batch_size={args.batch_size}",
            f"--lr={args.lr}",
            f"--num_nodes={args.num_nodes}",
        ]
        
        # 如果启用可视化，添加--visualize参数
        if args.visualize:
            fed_args.append("--visualize")
            
        # 如果启用高级态势感知，添加--advanced_awareness参数
        if args.advanced_awareness:
            fed_args.append("--advanced_awareness")
        
        # 将命令行参数转换为sys.argv格式
        sys.argv = ["federation/main.py"] + fed_args
        
        # 运行联邦学习主程序
        fed_main()
        
        # 最终内存清理
        print_memory_stats()
        
        # 如果生成了可视化图表，提示用户查看
        if args.visualize:
            vis_path = os.path.join(args.output_dir, "visualization")
            if os.path.exists(vis_path):
                print(f"\n可视化图表生成成功，存储在 {vis_path} 目录下")
                # 列出所有生成的图表
                print("\n生成的可视化图表:")
                for img_file in os.listdir(vis_path):
                    if img_file.endswith(".png"):
                        print(f" - {img_file}")
        
        print("联邦学习态势感知项目运行完成！")
    
    except Exception as e:
        print(f"程序运行出错: {e}")
        print_memory_stats()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 