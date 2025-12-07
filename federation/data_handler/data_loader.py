#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import json
import random
import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, random_split

class JADDDataset(Dataset):
    """JAAD数据集类 - 多模态版本"""
    
    def __init__(self, video_paths, annotations, transform=None, sequence_length=16):
        """
        初始化JAAD数据集
        
        Args:
            video_paths (list): 视频文件路径列表
            annotations (dict): 视频标注信息字典
            transform (callable, optional): 数据预处理转换函数
            sequence_length (int, optional): 视频序列长度
        """
        self.video_paths = video_paths
        self.annotations = annotations
        self.transform = transform
        self.sequence_length = sequence_length
        
        # 调试信息
        print(f"创建JAAD数据集: {len(video_paths)}个视频")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """获取一个数据样本"""
        try:
            video_path = self.video_paths[idx]
            video_id = os.path.basename(video_path).split('.')[0]
            
            # 加载视频帧
            frames = self._load_video_frames(video_path)
            
            # 获取标注
            annotation = self.annotations.get(video_id, {})
            
            # 获取多模态特征
            multimodal_features = self._extract_multimodal_features(video_id, annotation)
            
            # 获取行人动作标签 (简化为行人是否过马路: 1表示过马路, 0表示不过马路)
            label = 1 if annotation.get('crossing', False) else 0
            
            # 返回视频帧序列、多模态特征和标签
            return frames, multimodal_features, label
        except Exception as e:
            print(f"加载样本 {idx} 时出错: {e}")
            # 返回空数据，使用与_load_video_frames相同的图像大小
            empty_frames = torch.zeros((self.sequence_length, 3, 64, 64), dtype=torch.float32)
            empty_features = {
                'traffic': torch.zeros(5, dtype=torch.float32),  # 交通场景特征
                'vehicle': torch.zeros(4, dtype=torch.float32),  # 车辆行为特征
                'appearance': torch.zeros(8, dtype=torch.float32),  # 行人外观特征
                'attributes': torch.zeros(6, dtype=torch.float32)   # 行人属性特征
            }
            return empty_frames, empty_features, 0
    
    def _extract_multimodal_features(self, video_id, annotation):
        """
        提取多模态特征
        
        Args:
            video_id (str): 视频ID
            annotation (dict): 基本标注数据
        
        Returns:
            dict: 多模态特征字典
        """
        # 从annotations中提取基本信息
        # 注意：由于我们在加载过程中未读取全部XML文件，这里使用模拟数据
        # 在实际应用中，应根据实际读取的XML数据设置
        
        # 1. 交通场景特征
        traffic_features = torch.zeros(5, dtype=torch.float32)
        if annotation.get('traffic', {}):
            # One-hot编码道路类型: [市区, 郊区, 高速公路, 居民区, 其他]
            road_type = annotation['traffic'].get('road_type', 'urban')
            road_type_idx = {'urban': 0, 'suburban': 1, 'highway': 2, 'residential': 3, 'other': 4}.get(road_type, 4)
            traffic_features[road_type_idx] = 1.0
            
            # 是否存在人行横道
            traffic_features[1] = float(annotation['traffic'].get('ped_crossing', False))
            
            # 是否存在行人标志
            traffic_features[2] = float(annotation['traffic'].get('ped_sign', False))
            
            # 是否存在停车标志
            traffic_features[3] = float(annotation['traffic'].get('stop_sign', False))
            
            # 红绿灯状态 (无=0, 红=0.33, 黄=0.66, 绿=1.0)
            light_state = annotation['traffic'].get('traffic_light', 'none')
            light_value = {'none': 0.0, 'red': 0.33, 'yellow': 0.66, 'green': 1.0}.get(light_state, 0.0)
            traffic_features[4] = light_value
        
        # 2. 车辆行为特征
        vehicle_features = torch.zeros(4, dtype=torch.float32)
        if annotation.get('vehicle', {}):
            # 编码车辆状态: [慢速行驶, 减速, 停止, 加速]
            vehicle_state = annotation['vehicle'].get('state', 'none')
            if vehicle_state == 'moving_slow':
                vehicle_features[0] = 1.0
            elif vehicle_state == 'decelerating':
                vehicle_features[1] = 1.0
            elif vehicle_state == 'stopped':
                vehicle_features[2] = 1.0
            elif vehicle_state == 'accelerating':
                vehicle_features[3] = 1.0
        
        # 3. 行人外观特征
        appearance_features = torch.zeros(8, dtype=torch.float32)
        if annotation.get('appearance', {}):
            # 衣着颜色 (深色=0, 浅色=1)
            appearance_features[0] = 1.0 if annotation['appearance'].get('clothing_color', 'dark') == 'light' else 0.0
            
            # 是否携带包
            appearance_features[1] = float(annotation['appearance'].get('carrying_bag', False))
            
            # 是否携带背包
            appearance_features[2] = float(annotation['appearance'].get('carrying_backpack', False))
            
            # 是否戴帽子
            appearance_features[3] = float(annotation['appearance'].get('wearing_hat', False))
            
            # 是否戴墨镜
            appearance_features[4] = float(annotation['appearance'].get('wearing_sunglasses', False))
            
            # 朝向编码 (前=1, 后=2, 左=3, 右=4)
            pose = annotation['appearance'].get('pose', 'front')
            pose_value = {'front': 0.25, 'back': 0.5, 'left': 0.75, 'right': 1.0}.get(pose, 0.0)
            appearance_features[5] = pose_value
            
            # 是否有手势
            appearance_features[6] = float(annotation['appearance'].get('has_gesture', False))
            
            # 是否在交谈
            appearance_features[7] = float(annotation['appearance'].get('is_talking', False))
        
        # 4. 行人属性特征
        attributes_features = torch.zeros(6, dtype=torch.float32)
        if annotation.get('attributes', {}):
            # 年龄编码 (儿童=0, 青少年=0.33, 成人=0.66, 老人=1.0)
            age = annotation['attributes'].get('age', 'adult')
            age_value = {'child': 0.0, 'teenager': 0.33, 'adult': 0.66, 'senior': 1.0}.get(age, 0.66)
            attributes_features[0] = age_value
            
            # 性别编码 (女=0, 男=1)
            attributes_features[1] = 1.0 if annotation['attributes'].get('gender', 'male') == 'male' else 0.0
            
            # 群体大小 (归一化到0-1)
            group_size = min(5, int(annotation['attributes'].get('group_size', 1)))
            attributes_features[2] = (group_size - 1) / 4.0  # 1人映射到0, 5+人映射到1
            
            # 运动方向 (离开=0, 接近=1)
            attributes_features[3] = 1.0 if annotation['attributes'].get('motion_direction', 'away') == 'approaching' else 0.0
            
            # 道路上的位置 (人行道=0, 马路上=1)
            attributes_features[4] = 1.0 if annotation['attributes'].get('position', 'sidewalk') == 'road' else 0.0
            
            # 注意力状态 (未注意=0, 注意=1)
            attributes_features[5] = float(annotation['attributes'].get('attention', False))
        
        # 合并所有特征
        multimodal_features = {
            'traffic': traffic_features,
            'vehicle': vehicle_features,
            'appearance': appearance_features,
            'attributes': attributes_features
        }
        
        return multimodal_features
    
    def _load_video_frames(self, video_path):
        """加载视频帧"""
        # 创建空帧序列 - 注意这里维度顺序为[seq_len, channels, height, width]
        # 降低分辨率从112x112到64x64以减少内存使用
        img_size = 64
        frames = np.zeros((self.sequence_length, 3, img_size, img_size), dtype=np.float32)
        
        try:
            if not os.path.exists(video_path):
                # 如果视频不存在，返回一个全零的帧序列
                return torch.from_numpy(frames)
            
            # 设置OpenCV内存限制和参数
            # 降低读取缓冲区大小
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                return torch.from_numpy(frames)
            
            # 获取视频总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                cap.release()
                return torch.from_numpy(frames)
            
            # 采用较大的步长，减少需要处理的帧数
            step = max(1, total_frames // self.sequence_length)
            frame_indices = [min(total_frames-1, i*step) for i in range(self.sequence_length)]
            
            # 内存清理
            import gc
            gc.collect()
            
            for i, frame_idx in enumerate(frame_indices):
                try:
                    # 设置读取位置
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        # 关键优化：先缩小图像尺寸再进行其他处理，减少内存占用
                        frame = cv2.resize(frame, (img_size, img_size))
                        # BGR转RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # 归一化像素值至[0,1]
                        frame = frame.astype(np.float32) / 255.0
                        
                        # 交换维度顺序，从[height, width, channels]变为[channels, height, width]
                        frame = np.transpose(frame, (2, 0, 1))
                        
                        frames[i] = frame
                        
                        # 每处理3帧释放一次内存
                        if i % 3 == 0:
                            gc.collect()
                            
                except Exception as e:
                    print(f"处理帧 {frame_idx} 时出错: {e}")
                    # 出错时不终止，继续处理下一帧
                    # 清理内存后继续
                    gc.collect()
                    continue
            
            # 释放视频资源
            cap.release()
            
            # 再次清理内存
            gc.collect()
            
        except MemoryError:
            print(f"处理视频 {video_path} 时内存不足，返回降级版本帧")
            # 内存不足时返回一个低质量版本
            return torch.zeros((self.sequence_length, 3, img_size, img_size), dtype=torch.float32)
            
        except Exception as e:
            print(f"加载视频 {video_path} 时出错: {e}")
            # 发生其他错误时清理内存
            gc.collect()
        
        # 应用转换（如果有）
        if self.transform:
            frames = self.transform(frames)
        
        # 确保张量类型是正确的，并进行内存优化
        if isinstance(frames, np.ndarray):
            return torch.from_numpy(frames).contiguous()
        elif isinstance(frames, torch.Tensor):
            return frames.contiguous()
        else:
            # 处理其他类型的情况
            return torch.zeros((self.sequence_length, 3, img_size, img_size), dtype=torch.float32)

class JADDDataLoader:
    """JAAD数据加载器 - 多模态版本"""
    
    def __init__(self, data_path, annotation_path, batch_size=8, sequence_length=16):
        """
        初始化JAAD数据加载器
        
        Args:
            data_path (str): JAAD视频数据路径
            annotation_path (str): JAAD标注数据路径
            batch_size (int, optional): 批量大小
            sequence_length (int, optional): 视频序列长度
        """
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        
        # 加载视频文件路径
        self.video_paths = self._load_video_paths()
        
        # 加载所有类型的标注数据
        self.annotations = self._load_all_annotations()
        
        # 调试信息
        print(f"JAAD数据加载器初始化完成: {len(self.video_paths)}个视频")
    
    def _load_video_paths(self):
        """加载视频文件路径"""
        video_paths = []
        
        if os.path.exists(self.data_path):
            for filename in os.listdir(self.data_path):
                if filename.endswith('.mp4') or filename.endswith('.avi'):
                    video_path = os.path.join(self.data_path, filename)
                    video_paths.append(video_path)
        
        return video_paths
    
    def _load_all_annotations(self):
        """加载所有类型的标注数据"""
        # 主注释字典，按视频ID索引
        all_annotations = {}
        
        try:
            # 1. 加载基本标注
            basic_annotations = self._load_basic_annotations()
            
            # 2. 加载交通场景标注
            traffic_annotations = self._load_traffic_annotations()
            
            # 3. 加载车辆行为标注
            vehicle_annotations = self._load_vehicle_annotations()
            
            # 4. 加载行人外观标注
            appearance_annotations = self._load_appearance_annotations()
            
            # 5. 加载行人属性标注
            attribute_annotations = self._load_attribute_annotations()
            
            # 将所有标注合并到一个字典中
            for video_id in basic_annotations:
                all_annotations[video_id] = {
                    'basic': basic_annotations.get(video_id, {}),
                    'traffic': traffic_annotations.get(video_id, {}),
                    'vehicle': vehicle_annotations.get(video_id, {}),
                    'appearance': appearance_annotations.get(video_id, {}),
                    'attributes': attribute_annotations.get(video_id, {})
                }
                
                # 设置一些关键标志
                all_annotations[video_id]['crossing'] = basic_annotations.get(video_id, {}).get('crossing', False)
            
            print(f"成功加载了 {len(all_annotations)} 个视频的多模态标注数据")
            
        except Exception as e:
            print(f"加载所有标注时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return all_annotations
    
    def _load_basic_annotations(self):
        """加载基本标注数据"""
        annotations = {}
        
        # 尝试加载基本标注 (annotations/)
        path = os.path.join(self.annotation_path, 'annotations')
        if os.path.exists(path):
            try:
                # 尝试加载JSON格式的标注
                for filename in os.listdir(path):
                    if filename.endswith('.json'):
                        with open(os.path.join(path, filename), 'r') as f:
                            data = json.load(f)
                            annotations.update(data)
                    elif filename.endswith('.xml'):
                        video_id = filename.split('.')[0]
                        xml_path = os.path.join(path, filename)
                        
                        try:
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            
                            # 解析XML标注
                            ped_elements = root.findall(".//pedestrian")
                            crossing = any(ped.find("action/crossing") is not None for ped in ped_elements)
                            
                            annotations[video_id] = {
                                'crossing': crossing,
                                'pedestrians': len(ped_elements)
                            }
                        except Exception as e:
                            print(f"解析XML {xml_path} 时出错: {e}")
            except Exception as e:
                print(f"加载基本标注数据时出错: {e}")
        
        return annotations
    
    def _load_traffic_annotations(self):
        """加载交通场景标注数据"""
        annotations = {}
        
        # 尝试加载交通场景标注 (annotations_traffic/)
        path = os.path.join(self.annotation_path, 'annotations_traffic')
        if os.path.exists(path):
            try:
                for filename in os.listdir(path):
                    if filename.endswith('.xml'):
                        video_id = filename.split('_traffic')[0]
                        xml_path = os.path.join(path, filename)
                        
                        try:
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            
                            # 提取帧级别的交通信息
                            frames = root.findall(".//frame")
                            if frames:
                                # 使用最后一帧的交通信息（或任何代表性帧）
                                last_frame = frames[-1]
                                
                                road_type_elem = last_frame.find(".//road_type")
                                road_type = road_type_elem.text if road_type_elem is not None else "urban"
                                
                                ped_crossing = last_frame.find(".//ped_crossing") is not None
                                ped_sign = last_frame.find(".//ped_sign") is not None
                                stop_sign = last_frame.find(".//stop_sign") is not None
                                
                                traffic_light_elem = last_frame.find(".//traffic_light")
                                traffic_light = traffic_light_elem.text if traffic_light_elem is not None else "none"
                                
                                annotations[video_id] = {
                                    'road_type': road_type,
                                    'ped_crossing': ped_crossing,
                                    'ped_sign': ped_sign,
                                    'stop_sign': stop_sign,
                                    'traffic_light': traffic_light
                                }
                        except Exception as e:
                            print(f"解析交通XML {xml_path} 时出错: {e}")
            except Exception as e:
                print(f"加载交通场景标注数据时出错: {e}")
        
        return annotations
    
    def _load_vehicle_annotations(self):
        """加载车辆行为标注数据"""
        annotations = {}
        
        # 尝试加载车辆行为标注 (annotations_vehicle/)
        path = os.path.join(self.annotation_path, 'annotations_vehicle')
        if os.path.exists(path):
            try:
                for filename in os.listdir(path):
                    if filename.endswith('.xml'):
                        video_id = filename.split('_vehicle')[0]
                        xml_path = os.path.join(path, filename)
                        
                        try:
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            
                            # 提取帧级别的车辆信息
                            frames = root.findall(".//frame")
                            if frames:
                                # 使用最后一帧的车辆信息
                                last_frame = frames[-1]
                                
                                # 检测车辆状态
                                state = None
                                if last_frame.find(".//moving_slow") is not None:
                                    state = "moving_slow"
                                elif last_frame.find(".//decelerating") is not None:
                                    state = "decelerating"
                                elif last_frame.find(".//stopped") is not None:
                                    state = "stopped"
                                elif last_frame.find(".//accelerating") is not None:
                                    state = "accelerating"
                                
                                annotations[video_id] = {'state': state}
                        except Exception as e:
                            print(f"解析车辆XML {xml_path} 时出错: {e}")
            except Exception as e:
                print(f"加载车辆行为标注数据时出错: {e}")
        
        return annotations
    
    def _load_appearance_annotations(self):
        """加载行人外观标注数据"""
        annotations = {}
        
        # 尝试加载行人外观标注 (annotations_appearance/)
        path = os.path.join(self.annotation_path, 'annotations_appearance')
        if os.path.exists(path):
            try:
                for filename in os.listdir(path):
                    if filename.endswith('.xml'):
                        video_id = filename.split('_appearance')[0]
                        xml_path = os.path.join(path, filename)
                        
                        try:
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            
                            # 收集所有行人外观特征
                            ped_elements = root.findall(".//pedestrian")
                            if ped_elements:
                                # 使用第一个行人的外观特征
                                ped = ped_elements[0]
                                
                                clothing_color = "dark"
                                if ped.find(".//light_clothing") is not None:
                                    clothing_color = "light"
                                
                                carrying_bag = ped.find(".//carrying_bag") is not None
                                carrying_backpack = ped.find(".//carrying_backpack") is not None
                                wearing_hat = ped.find(".//wearing_hat") is not None
                                wearing_sunglasses = ped.find(".//wearing_sunglasses") is not None
                                
                                pose = "front"
                                if ped.find(".//pose_back") is not None:
                                    pose = "back"
                                elif ped.find(".//pose_left") is not None:
                                    pose = "left"
                                elif ped.find(".//pose_right") is not None:
                                    pose = "right"
                                
                                has_gesture = ped.find(".//gesture") is not None
                                is_talking = ped.find(".//talking") is not None
                                
                                annotations[video_id] = {
                                    'clothing_color': clothing_color,
                                    'carrying_bag': carrying_bag,
                                    'carrying_backpack': carrying_backpack,
                                    'wearing_hat': wearing_hat,
                                    'wearing_sunglasses': wearing_sunglasses,
                                    'pose': pose,
                                    'has_gesture': has_gesture,
                                    'is_talking': is_talking
                                }
                        except Exception as e:
                            print(f"解析外观XML {xml_path} 时出错: {e}")
            except Exception as e:
                print(f"加载行人外观标注数据时出错: {e}")
        
        return annotations
    
    def _load_attribute_annotations(self):
        """加载行人属性标注数据"""
        annotations = {}
        
        # 尝试加载行人属性标注 (annotations_attributes/)
        path = os.path.join(self.annotation_path, 'annotations_attributes')
        if os.path.exists(path):
            try:
                for filename in os.listdir(path):
                    if filename.endswith('.xml'):
                        video_id = filename.split('_attributes')[0]
                        xml_path = os.path.join(path, filename)
                        
                        try:
                            tree = ET.parse(xml_path)
                            root = tree.getroot()
                            
                            # 提取行人属性
                            ped_elements = root.findall(".//pedestrian")
                            if ped_elements:
                                # 使用第一个行人的属性
                                ped = ped_elements[0]
                                
                                age_elem = ped.find(".//age")
                                age = age_elem.text if age_elem is not None else "adult"
                                
                                gender_elem = ped.find(".//gender")
                                gender = gender_elem.text if gender_elem is not None else "male"
                                
                                group_size_elem = ped.find(".//group_size")
                                group_size = int(group_size_elem.text) if group_size_elem is not None else 1
                                
                                motion_direction = "away"
                                if ped.find(".//approaching") is not None:
                                    motion_direction = "approaching"
                                
                                position = "sidewalk"
                                if ped.find(".//road") is not None:
                                    position = "road"
                                
                                attention = ped.find(".//looking") is not None
                                
                                annotations[video_id] = {
                                    'age': age,
                                    'gender': gender,
                                    'group_size': group_size,
                                    'motion_direction': motion_direction,
                                    'position': position,
                                    'attention': attention
                                }
                        except Exception as e:
                            print(f"解析属性XML {xml_path} 时出错: {e}")
            except Exception as e:
                print(f"加载行人属性标注数据时出错: {e}")
        
        return annotations
    
    def _load_split_ids(self, split_name):
        """加载数据集划分ID"""
        split_path = os.path.join(self.annotation_path, 'split_ids', 'all_videos', f'{split_name}.txt')
        
        if not os.path.exists(split_path):
            print(f"警告: 找不到划分文件 {split_path}")
            return []
        
        with open(split_path, 'r') as f:
            video_ids = [line.strip() for line in f.readlines()]
        
        return video_ids
    
    def split_data(self, num_nodes=3):
        """
        划分数据集用于联邦学习
        
        Args:
            num_nodes (int): 边缘节点数量
        
        Returns:
            tuple: (训练数据集列表, 验证数据集, 测试数据集)
        """
        # 加载预定义的训练/验证/测试划分
        train_ids = self._load_split_ids('train')
        val_ids = self._load_split_ids('val')
        test_ids = self._load_split_ids('test')
        
        # 如果没有预定义划分，则随机划分
        if not train_ids or not val_ids or not test_ids:
            print("未找到预定义划分，使用随机划分...")
            all_paths = self.video_paths.copy()
            random.shuffle(all_paths)
            
            total = len(all_paths)
            val_size = int(total * 0.1)
            test_size = int(total * 0.1)
            train_size = total - val_size - test_size
            
            train_paths = all_paths[:train_size]
            val_paths = all_paths[train_size:train_size+val_size]
            test_paths = all_paths[train_size+val_size:]
        else:
            # 使用预定义划分
            train_paths = [p for p in self.video_paths if os.path.basename(p).split('.')[0] in train_ids]
            val_paths = [p for p in self.video_paths if os.path.basename(p).split('.')[0] in val_ids]
            test_paths = [p for p in self.video_paths if os.path.basename(p).split('.')[0] in test_ids]
        
        # 创建验证和测试数据集
        val_dataset = JADDDataset(val_paths, self.annotations, sequence_length=self.sequence_length)
        test_dataset = JADDDataset(test_paths, self.annotations, sequence_length=self.sequence_length)
        
        # 为每个节点划分训练数据
        random.shuffle(train_paths)
        node_train_datasets = []
        
        # 计算每个节点的数据量
        paths_per_node = len(train_paths) // num_nodes
        
        for i in range(num_nodes):
            # 最后一个节点可能会有额外的数据
            if i == num_nodes - 1:
                node_paths = train_paths[i*paths_per_node:]
            else:
                node_paths = train_paths[i*paths_per_node:(i+1)*paths_per_node]
            
            node_dataset = JADDDataset(node_paths, self.annotations, sequence_length=self.sequence_length)
            node_train_datasets.append(node_dataset)
            
            print(f"节点{i}分配的数据量: {len(node_paths)}个视频")
        
        return node_train_datasets, val_dataset, test_dataset
    
    def get_dataloader(self, dataset, batch_size=None, shuffle=True):
        """
        获取数据加载器
        
        Args:
            dataset: 数据集对象
            batch_size (int, optional): 批量大小
            shuffle (bool, optional): 是否打乱数据
        
        Returns:
            DataLoader: PyTorch数据加载器
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # 减少worker数量，避免内存问题
            pin_memory=False  # 不使用pin_memory以减少内存使用
        ) 