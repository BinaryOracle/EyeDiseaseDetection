# 眼底图片疾病检测

这是一个关于眼底图片疾病检测模型的项目，基于 [RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE) 的模型实现进行迁移学习和微调，以实现对眼底图片中疾病的检测和分类。

## 项目介绍

本项目利用深度学习技术，特别是基于MAE（Masked Autoencoder）的RETFound模型，对眼底图片进行疾病检测和分类。通过迁移学习和微调预训练模型，项目能够识别多种眼底疾病，包括AMD、CSC、DR、RP、RVO等。

## 项目结构

- `config`: 存放常用的命令行参数配置
- `data`: 存放数据集
- `diagnose_output`: 存放诊断结果输出文件
- `output`: 存放基于预训练模型微调得到的检查点
- `pretrain`: 存放预训练模型检查点
- `config.py`: 存放通用配置
- `diagnose.py`: 存放诊断脚本
- `finetune.py`: 存放微调脚本
- `mockClient.py`: 存放模拟实现的消息队列客户端
- `models_vit.py`: 存放RETFound_MAE模型实现
- `predict.py`: 存放预测脚本
- `training_config.py`: 训练配置文件，包含类别不平衡处理策略
- `analyze_imbalance.py`: 类别不平衡分析脚本

## 数据集分析

### 类别分布统计

根据 `config/data.json` 的统计信息，当前数据集包含以下类别：

| 类别代码 | 类别名称 | 样本数量 | 占比 |
|---------|---------|---------|------|
| D | 糖尿病性视网膜病变 | 39,314 | 48.6% |
| D-level | 糖尿病性视网膜病变(分级) | 38,788 | 47.9% |
| A | 年龄相关性黄斑变性 | 863 | 1.1% |
| RVO | 视网膜静脉阻塞 | 491 | 0.6% |
| G | 青光眼 | 430 | 0.5% |
| C | 白内障 | 424 | 0.5% |
| N | 正常眼底 | 245 | 0.3% |
| CSC | 中央性浆液性脉络膜视网膜病变 | 235 | 0.3% |
| RP | 视网膜色素变性 | 158 | 0.2% |

**总计**: 80,948 张图片，9个类别

### 类别不平衡问题分析

通过 `analyze_imbalance.py` 脚本分析，发现严重的类别不平衡问题：

- **不平衡比例**: 248.8:1 (D类39,314张 vs RP类158张)
- **Gini系数**: 0.751 (高度不平等)
- **严重程度**: **严重** (severe)
- **Shannon熵**: 1.276 (低信息熵)
- **标准化熵**: 0.403 (远低于理想值1.0)

## 类别不平衡优化方案

### 问题描述

数据集存在严重的类别不平衡问题，主要类别(D和D-level)占据了96.5%的样本，而其他7个类别仅占3.5%。这种不平衡会导致：

1. 模型偏向于预测主要类别
2. 小类别识别准确率低
3. 训练过程不稳定
4. 泛化能力差

### 优化策略

#### 1. 加权损失函数 (Weighted Loss)
```python
# 使用sklearn自动计算类别权重
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(all_labels), 
    y=all_labels
)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

#### 2. 加权随机采样 (Weighted Random Sampling)
```python
# 为训练集创建加权采样器
train_sample_weights = [1.0 / train_label_counts[label] for label in train_labels]
train_sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_labels),
    replacement=True
)
```

#### 3. 增强数据增强 (Enhanced Data Augmentation)
```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),           # 随机旋转
    transforms.ColorJitter(                  # 颜色抖动
        brightness=0.2, contrast=0.2, 
        saturation=0.2, hue=0.1
    ),
    transforms.Resize(255),
    transforms.CenterCrop(255),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### 4. 学习率调度 (Learning Rate Scheduling)
```python
# 使用余弦退火重启调度器
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.01
)
```

#### 5. 梯度裁剪 (Gradient Clipping)
```python
# 防止梯度爆炸，提高训练稳定性
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 6. 分层采样 (Stratified Sampling)
```python
# 训练/验证集划分保持类别比例
train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
    all_img_paths, all_labels, 
    test_size=args.val_ratio, 
    random_state=42,
    stratify=all_labels  # 分层采样
)
```

### 推荐训练参数

针对严重类别不平衡问题，推荐以下训练参数：

```python
# 基础参数
batch_size = 16          # 减小批次大小，提高稳定性
lr = 0.001              # 降低学习率，避免过拟合
weight_decay = 0.1      # 增加正则化强度
epochs = 100            # 增加训练轮数

# 优化策略
use_weighted_loss = True           # 启用加权损失
use_weighted_sampling = True       # 启用加权采样
use_data_augmentation = True       # 启用数据增强
use_gradient_clipping = True       # 启用梯度裁剪
use_learning_rate_scheduling = True # 启用学习率调度
```

### 训练流程改进

#### 实时监控
- 每个epoch显示各类别准确率
- 跟踪训练损失变化趋势
- 显示学习率变化
- 详细的训练进度统计

#### 性能评估
```python
# 计算每个类别的准确率
def calculate_class_metrics(outputs, labels, num_classes):
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels)
    
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    for i in range(num_classes):
        mask = (labels == i)
        class_correct[i] = correct[mask].sum()
        class_total[i] = mask.sum()
    
    return class_correct, class_total
```

## 微调过程

1. 准备数据集：将眼底图片按照疾病类型分类存放在`data`目录下对应的子目录中。

2. 下载预训练模型检查点：从[HuggingFace](https://huggingface.co)下载RETFound预训练模型权重，并放置在`pretrain`目录下(本项目训练采用的是私有非公开的权重文件)。

3. 运行微调脚本：

```bash
# 使用优化后的参数
python finetune.py \
    --data_path ./data \
    --model RETFound_mae \
    --nb_classes 8 \
    --batch_size 16 \
    --lr 0.001 \
    --epochs 100 \
    --save_path ./output
```

4. 微调完成后，最佳模型权重将保存在`output/checkpoint-best.pth`。

### 微调细节

图片增强: 训练时采用水平、垂直随机翻转，随机旋转，颜色抖动，中心裁剪加归一化，并调整图像大小为 `255x255` 分辨率。 测试时移除了随机增强操作，仅保留中心裁剪和归一化。

预训练模型权重加载: 移除分类数量不匹配的分类头，替换为新的分类头，并对分类头的权重进行截断正态分布初始化，并将偏置项初始化为零。

微调方法:  Adapter Tuning（适配器微调）。

```python
Input ──> LN ──> Attention ──> Add ──> LN ──> MLP ──> Add ──> Output
                  │                          │
                  ▼                          ▼
             Adapter-1                  Adapter-2
```

## 结果

基于官方开源的 [YukunZhou/RETFound_mae_natureCFP](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) 检查点执行 liner probe 微调的准确率在 82% 左右。

基于私有数据集进行训练得到的检查点版本的准确率最高约为 89.3%。

**优化后预期效果**:
- 缓解类别不平衡问题
- 提高小类别识别准确率
- 稳定训练过程
- 改善模型泛化能力

## 诊断过程

1. 从消息队列异步拉取患者信息,视力检测信息等，包括:

```json
"Patient": {
  "patient_id": "患者唯一ID",
  "name": "姓名",
  "age": "年龄",
  "gender": "性别",
  "phone": "电话",
  "photos": "患者眼底图片列表",
  "screening_records": "患者历次检查记录"
}

"vision_data": {
  "left_eye": 0.8,
  "right_eye": 0.6,
  "intraocular_pressure": {"left": 18, "right": 19}
}
```

2. 基于微调好的检查点对眼底图片进行疾病检测,并返回检测结果和置信度:

```json
{
  "conclusion": "高度近视",
  "confidence": 0.89,
  "abnormal_areas": ["视网膜周边"]
}
```

3. 医师结合AI审核结果，给出最终的诊断建议:

```json
{
  "final_conclusion": "高度近视",
  "doctor_name": "唐医生",
  "recommendation": "3个月后复查，避免剧烈运动"
}
```

4. 结合大语言模型如: chatgpt,deepseek等，综合分析患者情况，AI诊断结果，医师诊断建议,最终生成一份专业的诊断报告:

```json
{
  "patients": {
    "P2025001": {
      "patient_id": "P2025001",
      "name": "扎西",
      "age": 45,
      "gender": "男",
      "phone": "13800138000",
      "screening_records": [
        {
          "screening_time": "2025-07-31 16:09:07",
          "eye_image_path": "/Users/zhandaohong/PycharmProjects/EyeDiseaseDetection/data/A-CFP/CAI_CHENGHUI_19460619_20201209_1910_IMAGEnetR4_Image_OD_1.2.392.200106.1651.4.2.200217210022131239.1607483470.153.tiff",
          "vision_data": {
            "left_eye": 0.8,
            "right_eye": 0.6,
            "intraocular_pressure": {
              "left": 18,
              "right": 19
            }
          },
          "notes": "",
          "ai_diagnosis": {
            "conclusion": "A-CFP",
            "confidence": 0.6787625551223755,
            "abnormal_areas": [
              "视网膜周边"
            ]
          },
          "doctor_review": {
            "final_conclusion": "高度近视",
            "doctor_name": "唐医生",
            "recommendation": "3个月后复查，避免剧烈运动"
          },
          "report_generated": true
        }
      ]
    }
  },
  "ai_diagnosis_results": {
    "P2025001_0": {
      "conclusion": "A-CFP",
      "confidence": 0.6787625551223755,
      "abnormal_areas": [
        "视网膜周边"
      ]
    }
  },
  "save_time": "2025-07-31 16:09:07"
}
``` 

5. 将诊断结果信息投递到消息队列，最终会在小程序界面上展示给用户观看。

## 使用说明

### 类别不平衡分析

运行分析脚本查看详细的类别分布和不平衡程度：

```bash
python analyze_imbalance.py
```

该脚本会生成：
- 详细的类别不平衡分析报告
- 可视化图表 (`class_imbalance_analysis.png`)
- 针对不平衡程度的训练建议

### 训练配置

使用 `training_config.py` 文件管理训练参数：

```python
from training_config import get_training_config, get_recommended_config_for_imbalance

# 获取基础配置
config = get_training_config()

# 根据不平衡分析获取推荐配置
recommended = get_recommended_config_for_imbalance(imbalance_analysis)
```

### 注意事项

1. **数据准备**: 确保各类别数据分布合理，避免极端不平衡
2. **参数调优**: 根据不平衡程度调整批次大小和学习率
3. **监控训练**: 密切关注各类别的准确率变化
4. **模型评估**: 使用加权准确率、F1分数等指标评估模型性能

## 技术栈

- **深度学习框架**: PyTorch
- **模型架构**: RETFound_MAE (Vision Transformer)
- **数据处理**: PIL, torchvision
- **数据增强**: 随机翻转、旋转、颜色抖动
- **优化器**: AdamW
- **学习率调度**: CosineAnnealingWarmRestarts
- **类别不平衡处理**: 加权损失、加权采样、数据增强

