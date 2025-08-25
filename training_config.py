#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练配置文件 - 针对类别不平衡问题优化
"""

import torch
import numpy as np

# 基础训练参数
TRAINING_CONFIG = {
    # 模型参数
    'model': 'RETFound_mae',
    'input_size': 255,
    'nb_classes': 8,
    'global_pool': 'token',
    'drop_path': 0.2,
    
    # 训练参数
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.002,
    'weight_decay': 0.05,
    'val_ratio': 0.2,
    
    # 类别不平衡处理参数
    'use_weighted_loss': True,
    'use_weighted_sampling': True,
    'use_data_augmentation': True,
    'use_gradient_clipping': True,
    'use_learning_rate_scheduling': True,
    
    # 数据增强参数
    'augmentation': {
        'random_rotation': 10,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        },
        'random_horizontal_flip': True,
        'random_vertical_flip': True
    },
    
    # 学习率调度参数
    'lr_scheduler': {
        'type': 'cosine_annealing_warm_restarts',
        'T_0': 10,
        'T_mult': 2,
        'eta_min_factor': 0.01
    },
    
    # 梯度裁剪参数
    'gradient_clipping': {
        'max_norm': 1.0
    }
}

# 类别权重计算策略
CLASS_WEIGHT_STRATEGIES = {
    'balanced': 'balanced',  # sklearn的balanced策略
    'inverse': 'inverse',    # 1/样本数
    'sqrt_inverse': 'sqrt_inverse',  # 1/sqrt(样本数)
    'custom': 'custom'       # 自定义权重
}

# 自定义类别权重（基于数据分布分析）
CUSTOM_CLASS_WEIGHTS = {
    'A': 1.0,      # AMD - 863张
    'C': 1.0,      # Cataract - 424张  
    'CSC': 1.5,    # CSC - 235张
    'D': 0.1,      # Diabetes - 39314张（主要类别，权重降低）
    'G': 1.0,      # Glaucoma - 430张
    'N': 2.0,      # Normal - 245张
    'RP': 2.5,     # RP - 158张
    'RVO': 1.5     # RVO - 491张
}

def get_class_weights(strategy='balanced', class_names=None):
    """
    获取类别权重
    
    Args:
        strategy: 权重策略
        class_names: 类别名称列表
    
    Returns:
        torch.Tensor: 类别权重
    """
    if strategy == 'custom' and class_names is not None:
        weights = []
        for class_name in class_names:
            weights.append(CUSTOM_CLASS_WEIGHTS.get(class_name, 1.0))
        return torch.FloatTensor(weights)
    
    return None

def get_training_config():
    """获取训练配置"""
    return TRAINING_CONFIG.copy()

def get_augmentation_config():
    """获取数据增强配置"""
    return TRAINING_CONFIG['augmentation'].copy()

def get_lr_scheduler_config():
    """获取学习率调度器配置"""
    return TRAINING_CONFIG['lr_scheduler'].copy()

# 类别不平衡分析
def analyze_class_imbalance(class_counts):
    """
    分析类别不平衡程度
    
    Args:
        class_counts: 各类别样本数量字典
    
    Returns:
        dict: 不平衡分析结果
    """
    counts = list(class_counts.values())
    total = sum(counts)
    
    # 计算不平衡指标
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # 计算Gini系数
    sorted_counts = sorted(counts)
    n = len(counts)
    cumsum = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    # 计算Shannon熵
    proportions = [c/total for c in counts if c > 0]
    entropy = -sum(p * np.log2(p) for p in proportions)
    max_entropy = np.log2(len(counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        'total_samples': total,
        'num_classes': len(counts),
        'max_count': max_count,
        'min_count': min_count,
        'imbalance_ratio': imbalance_ratio,
        'gini_coefficient': gini,
        'shannon_entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'severity': 'severe' if imbalance_ratio > 100 else 'moderate' if imbalance_ratio > 10 else 'mild'
    }

# 推荐配置
def get_recommended_config_for_imbalance(imbalance_analysis):
    """
    根据不平衡分析结果推荐配置
    
    Args:
        imbalance_analysis: 不平衡分析结果
    
    Returns:
        dict: 推荐的配置
    """
    severity = imbalance_analysis['severity']
    
    if severity == 'severe':
        return {
            'use_weighted_loss': True,
            'use_weighted_sampling': True,
            'use_data_augmentation': True,
            'batch_size': 16,  # 减小批次大小
            'lr': 0.001,       # 降低学习率
            'weight_decay': 0.1,  # 增加正则化
            'use_gradient_clipping': True,
            'use_learning_rate_scheduling': True
        }
    elif severity == 'moderate':
        return {
            'use_weighted_loss': True,
            'use_weighted_sampling': True,
            'use_data_augmentation': True,
            'batch_size': 24,
            'lr': 0.0015,
            'weight_decay': 0.05,
            'use_gradient_clipping': True,
            'use_learning_rate_scheduling': True
        }
    else:  # mild
        return {
            'use_weighted_loss': False,
            'use_weighted_sampling': False,
            'use_data_augmentation': True,
            'batch_size': 32,
            'lr': 0.002,
            'weight_decay': 0.05,
            'use_gradient_clipping': False,
            'use_learning_rate_scheduling': True
        }
