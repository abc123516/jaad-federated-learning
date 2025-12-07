# 基于JAAD数据集的多模态联邦学习态势感知项目

## 项目简介

本项目基于JAAD（Joint Attention in Autonomous Driving）数据集，通过横向联邦学习策略训练行人行为预测模型，利用多模态数据融合提高预测准确性。同时，模拟边缘计算场景下的态势感知，包括节点故障处理、动态权重调整及预测结果融合。

## 项目架构

```
federation/
  ├── __init__.py              # 包初始化文件
  ├── main.py                  # 项目主程序
  ├── models/                  # 模型模块
  │   ├── __init__.py
  │   ├── resnet_gru.py        # ResNet-50 + GRU 模型 (节点1)
  │   ├── mobilenet_lstm.py    # MobileNetV3 + LSTM 模型 (节点2)
  │   ├── efficientnet_transformer.py  # EfficientNet + Transformer 模型 (节点3)
  │   └── multimodal_fusion.py # 多模态融合模型 (节点4)
  ├── edge_nodes/              # 边缘节点模块
  │   ├── __init__.py
  │   └── node_manager.py      # 边缘节点管理器
  ├── data_handler/            # 数据处理模块
  │   ├── __init__.py
  │   └── data_loader.py       # JAAD多模态数据集加载器
  └── utils/                   # 工具模块
      ├── __init__.py
      ├── fed_aggregator.py    # 联邦聚合器
      ├── situation_awareness.py  # 基础态势感知模块
      └── advanced_situation_awareness.py  # 高级态势感知模块（多模态融合与不确定性量化）
```

## 技术特点

1. **横向联邦学习**: 数据分布在不同边缘节点，模型在本地训练后通过参数聚合形成全局模型
2. **异构模型架构**: 四种不同的模型架构分别部署在不同边缘节点
   - 节点1: ResNet-50 + GRU (视觉模态)
   - 节点2: MobileNetV3 + LSTM (视觉模态)
   - 节点3: EfficientNet + Transformer (视觉模态)
   - 节点4: 多模态融合模型 (视觉+交通场景+车辆行为+行人外观与属性)
3. **多模态数据融合**: 融合五种不同模态的数据
   - 视觉特征 (通过MobileNetV2+GRU提取)
   - 交通场景特征 (道路类型、人行横道、交通信号灯等)
   - 车辆行为特征 (加速、减速、停止等)
   - 行人外观特征 (衣着颜色、携带物品、姿势朝向等)
   - 行人属性特征 (年龄、性别、群体大小等)
4. **高级态势感知**: 基于贝叶斯理论和核函数模型的态势感知
   - 核函数多模态数据融合
   - 模型不确定性量化 (基于信息熵)
   - 自适应强化学习动态权重调整
   - 实时性能监控与可视化
5. **故障处理与优化**: 
   - 节点故障自动检测与任务迁移
   - 基于性能指标的动态聚合权重调整
   - 预测结果融合与不确定性分析

## 环境要求

- Python 3.7+
- PyTorch 1.8.0+
- OpenCV 4.5.0+
- NumPy 1.19.0+
- Matplotlib 3.3.0+
- SciPy 1.6.0+
- JAAD数据集

## 安装与使用

1. 克隆代码库

```bash
git clone https://github.com/your-username/jaad-federated-learning.git
cd jaad-federated-learning
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 准备数据集

将JAAD数据集放置在指定目录：
- 视频数据：`./JAAD_clips/`
- 标注数据：`./JAAD-JAAD_2.0/JAAD-JAAD_2.0/`

4. 运行项目

```bash
# 使用基础态势感知
python run.py --epochs 50 --num_nodes 4

# 使用高级态势感知(多模态融合与不确定性量化)
python run.py --epochs 5 --num_nodes 4 --advanced_awareness
```

## 参数说明

- `--data_path`: JAAD视频数据路径
- `--annotation_path`: JAAD标注数据路径
- `--epochs`: 联邦学习总轮数
- `--local_epochs`: 本地训练轮数
- `--batch_size`: 批量大小
- `--lr`: 学习率
- `--num_nodes`: 边缘节点数量
- `--output_dir`: 输出目录
- `--visualize`: 是否生成可视化图表
- `--advanced_awareness`: 是否使用高级态势感知模块

## 项目流程

1. **多模态数据处理**:
   - 视频帧提取与预处理
   - 多模态标注信息解析 (交通场景、车辆行为、行人外观和属性)
   - 特征融合前的归一化与变换

2. **模型训练流程**:
   - 本地训练: 各节点独立训练本地模型
   - 态势感知: 评估各节点性能与不确定性
   - 权重调整: 基于态势感知结果动态调整聚合权重
   - 参数聚合: 中央服务器聚合模型参数
   - 模型更新: 将聚合后的参数分发至各节点

3. **高级态势感知**:
   - 计算核函数值处理多模态数据
   - 通过信息熵H(y)量化预测不确定性
   - 基于准确率和不确定性自适应调整模态权重w_i(t)
   - 生成多模态vs单模态性能对比报告
   - 实时可视化模态权重分布与性能指标

4. **故障处理与优化**:
   - 自动检测节点异常
   - 任务动态迁移策略
   - 性能数据持续监控与记录

## 多模态融合示例

多模态融合模型架构:

```
视频帧序列 → MobileNetV2 → GRU → 视觉特征向量
                                      ↓
交通场景特征 → 线性投影 →              ↓
车辆行为特征 → 线性投影 → 自注意力融合 → 分类器 → 预测结果
行人外观特征 → 线性投影 →              ↓
行人属性特征 → 线性投影 →              ↓
```
