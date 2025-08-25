#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析类别不平衡问题并生成训练建议
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from training_config import analyze_class_imbalance, get_recommended_config_for_imbalance

def load_data_stats():
    """加载数据统计信息"""
    try:
        with open('config/data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("❌ 找不到 config/data.json 文件")
        return None

def analyze_class_distribution():
    """分析类别分布"""
    data = load_data_stats()
    if not data:
        return
    
    print("🔍 类别不平衡分析报告")
    print("=" * 80)
    
    # 提取类别信息
    categories = data['categories']
    class_counts = {}
    
    print("📊 各类别样本数量:")
    for category in categories:
        code = category['code']
        total = category['counts']['total']
        class_counts[code] = total
        print(f"  {code} ({category['name']}): {total:,} 张图片")
    
    print(f"\n📈 总样本数: {data['statistics']['total_images']:,}")
    print(f"🏷️  类别数: {len(categories)}")
    
    # 分析不平衡程度
    imbalance_analysis = analyze_class_imbalance(class_counts)
    
    print("\n⚖️  不平衡程度分析:")
    print(f"  最大类别样本数: {imbalance_analysis['max_count']:,}")
    print(f"  最小类别样本数: {imbalance_analysis['min_count']:,}")
    print(f"  不平衡比例: {imbalance_analysis['imbalance_ratio']:.1f}:1")
    print(f"  Gini系数: {imbalance_analysis['gini_coefficient']:.3f}")
    print(f"  Shannon熵: {imbalance_analysis['shannon_entropy']:.3f}")
    print(f"  标准化熵: {imbalance_analysis['normalized_entropy']:.3f}")
    print(f"  不平衡严重程度: {imbalance_analysis['severity']}")
    
    # 计算各类别占比
    total = sum(class_counts.values())
    print(f"\n📊 各类别占比:")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for code, count in sorted_classes:
        percentage = (count / total) * 100
        print(f"  {code}: {percentage:.1f}% ({count:,} 张)")
    
    # 获取推荐配置
    recommended_config = get_recommended_config_for_imbalance(imbalance_analysis)
    
    print(f"\n💡 针对{imbalance_analysis['severity']}不平衡的推荐配置:")
    for key, value in recommended_config.items():
        print(f"  {key}: {value}")
    
    # 生成可视化图表
    generate_visualization(class_counts, imbalance_analysis)
    
    return class_counts, imbalance_analysis, recommended_config

def generate_visualization(class_counts, imbalance_analysis):
    """生成可视化图表"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('类别不平衡分析可视化', fontsize=16, fontweight='bold')
        
        # 1. 柱状图 - 各类别样本数量
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax1.bar(classes, counts, color='skyblue', alpha=0.7)
        ax1.set_title('各类别样本数量分布')
        ax1.set_xlabel('类别')
        ax1.set_ylabel('样本数量')
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱子上添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # 2. 饼图 - 各类别占比
        total = sum(counts)
        percentages = [count/total*100 for count in counts]
        
        # 只显示占比大于1%的类别，其他归为"其他"
        threshold = 1.0
        significant_classes = []
        significant_percentages = []
        other_percentage = 0
        
        for i, (cls, pct) in enumerate(zip(classes, percentages)):
            if pct >= threshold:
                significant_classes.append(cls)
                significant_percentages.append(pct)
            else:
                other_percentage += pct
        
        if other_percentage > 0:
            significant_classes.append('其他')
            significant_percentages.append(other_percentage)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(significant_classes)))
        wedges, texts, autotexts = ax2.pie(significant_percentages, labels=significant_classes, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('各类别占比分布')
        
        # 3. 对数坐标图 - 更好地显示不平衡
        ax3.bar(classes, counts, color='lightcoral', alpha=0.7)
        ax3.set_title('各类别样本数量分布 (对数坐标)')
        ax3.set_xlabel('类别')
        ax3.set_ylabel('样本数量 (对数)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')
        
        # 4. 不平衡指标总结
        ax4.axis('off')
        summary_text = f"""
不平衡分析总结:

最大类别: {imbalance_analysis['max_count']:,} 张
最小类别: {imbalance_analysis['min_count']:,} 张
不平衡比例: {imbalance_analysis['imbalance_ratio']:.1f}:1
Gini系数: {imbalance_analysis['gini_coefficient']:.3f}
严重程度: {imbalance_analysis['severity']}

建议策略:
• 使用加权损失函数
• 实施数据增强
• 采用分层采样
• 调整学习率策略
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('class_imbalance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 可视化图表已保存为: class_imbalance_analysis.png")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"⚠️  生成可视化图表时出错: {e}")
        print("请确保已安装 matplotlib 库")

def generate_training_recommendations(class_counts, imbalance_analysis):
    """生成训练建议"""
    print(f"\n🎯 训练策略建议")
    print("=" * 80)
    
    severity = imbalance_analysis['severity']
    
    if severity == 'severe':
        print("🚨 检测到严重类别不平衡，建议采取以下措施:")
        print("   1. 使用加权交叉熵损失函数")
        print("   2. 实施加权随机采样")
        print("   3. 增强数据增强策略")
        print("   4. 降低学习率 (建议: 0.001)")
        print("   5. 减小批次大小 (建议: 16)")
        print("   6. 增加正则化强度")
        print("   7. 使用梯度裁剪")
        print("   8. 实施学习率调度")
        
    elif severity == 'moderate':
        print("⚠️  检测到中等类别不平衡，建议采取以下措施:")
        print("   1. 使用加权交叉熵损失函数")
        print("   2. 实施加权随机采样")
        print("   3. 适度数据增强")
        print("   4. 调整学习率 (建议: 0.0015)")
        print("   5. 使用梯度裁剪")
        print("   6. 实施学习率调度")
        
    else:
        print("✅ 类别不平衡程度较轻，建议采取以下措施:")
        print("   1. 适度数据增强")
        print("   2. 使用学习率调度")
        print("   3. 监控各类别性能")
    
    print(f"\n📋 具体参数建议:")
    print(f"   批次大小: {16 if severity == 'severe' else 24 if severity == 'moderate' else 32}")
    print(f"   学习率: {0.001 if severity == 'severe' else 0.0015 if severity == 'moderate' else 0.002}")
    print(f"   权重衰减: {0.1 if severity == 'severe' else 0.05}")
    print(f"   梯度裁剪: {'是' if severity in ['severe', 'moderate'] else '否'}")

if __name__ == "__main__":
    print("🔍 开始分析类别不平衡问题...")
    class_counts, imbalance_analysis, recommended_config = analyze_class_distribution()
    
    if class_counts:
        generate_training_recommendations(class_counts, imbalance_analysis)
        
        print(f"\n💾 分析完成！建议根据上述配置优化训练流程。")
        print(f"📁 详细配置请参考: training_config.py")
