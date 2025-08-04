import torch
from torchvision import transforms
import config
from PIL import Image

class_tabel = ('AMD-CFP', 'CSC-CFP', 'DR-CFP', 'normal-CFP', 'RP-CFP-CFP', 'RVO-CFP')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 预测函数
def predict(img_path, model, args):
    # 从路径加载图像并转换为RGB格式
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # 添加批次维度
    img_tensor = img_tensor.to(config.device).float()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        topk_probs, topk_indices = torch.topk(probs, k=args.nb_classes)
        if topk_probs[0].item() > 0.5:  # 如果预测出的概率不足0.5则认为这张图片分类失败
            return class_tabel[topk_indices[0].item()], topk_probs[0].item()
        else:
            return "Uncertain", topk_probs[0].item()