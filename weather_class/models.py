import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPVisionModel,AutoImageProcessor,AutoModel



class DinoV3Classifier(nn.Module):
    """DINOv3 ConvNeXt 分类模型（使用ModelScope）"""
    def __init__(self, num_classes=6, pretrained_path='dinov3-convnext-tiny-pretrain-lvd1689m'):
        super(DinoV3Classifier, self).__init__()
        

        
        print(f"加载DINOv3模型: {pretrained_path}")
        
        # 加载DINOv3预训练模型（使用ModelScope）
        self.processor = AutoImageProcessor.from_pretrained(pretrained_path)
        self.backbone = AutoModel.from_pretrained(pretrained_path)
        
        # 冻结backbone参数（可选）
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 获取特征维度（ConvNeXt-Tiny的输出维度是768）
        self.feature_dim = 768
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"DINOv3模型加载完成，特征维度: {self.feature_dim}")
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，形状为(B, C, H, W)，已经过标准化
        """
        # 将标准化的tensor转换回[0, 1]范围
        # 因为processor期望[0, 1]范围的输入
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_denorm = x * std + mean
        
        # 使用processor处理（这会再次进行标准化，但使用DINOv3的标准化参数）
        # 注意：这里我们直接传入pixel_values，跳过processor的处理
        # 因为processor期望PIL图像，但我们已经有tensor了
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x_denorm)
        
        # 使用池化后的特征
        pooled_output = outputs.pooler_output
        
        # 分类
        output = self.classifier(pooled_output)
        return output


class CLIPViTClassifier(nn.Module):
    """CLIP ViT 分类模型"""
    def __init__(self, num_classes=6, model_path='clip-vit-base-patch32'):
        super(CLIPViTClassifier, self).__init__()
        
        # 加载CLIP的视觉编码器
        self.backbone = CLIPVisionModel.from_pretrained(model_path)
        
        #冻结backbone参数（可选）
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 分类头（CLIP ViT-B/32的输出维度是768）
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # CLIP输出
        outputs = self.backbone(pixel_values=x)
        # 使用池化后的特征
        pooled_output = outputs.pooler_output
        # 分类
        output = self.classifier(pooled_output)
        return output


class VitClassifier(nn.Module):
    """DINOv3 ConvNeXt 分类模型（使用ModelScope）"""

    def __init__(self, num_classes=6, pretrained_path='vit-base-patch16-224'):
        super(VitClassifier, self).__init__()

        self.processor = AutoImageProcessor.from_pretrained(pretrained_path)
        self.backbone = AutoModel.from_pretrained(pretrained_path)


        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_dim = 768

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入tensor，形状为(B, C, H, W)，已经过标准化
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_denorm = x * std + mean

        with torch.no_grad():
            outputs = self.backbone(pixel_values=x_denorm)

        # 使用池化后的特征
        pooled_output = outputs.pooler_output

        # 分类
        output = self.classifier(pooled_output)
        return output


class ResNet34Classifier(nn.Module):
    """ResNet34 分类模型"""
    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet34Classifier, self).__init__()
        
        # 加载预训练的ResNet34
        self.backbone = models.resnet34(pretrained=pretrained)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class ResNet152Classifier(nn.Module):

    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet152Classifier, self).__init__()

        # 加载预训练的ResNet101
        self.backbone = models.resnet152(pretrained=pretrained)

        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def get_model(model_name, num_classes=6, **kwargs):
    """
    获取指定的模型
    
    Args:
        model_name: 模型名称 ('dinov3', 'clip', 'resnet34')
        num_classes: 分类数目
        **kwargs: 其他参数
    
    Returns:
        model: 对应的模型
    """
    if model_name == 'dinov3':
        pretrained_path = kwargs.get('pretrained_path', './dinov3-vitb16-pretrain-lvd1689m')
        return DinoV3Classifier(num_classes=num_classes, pretrained_path=pretrained_path)
    
    elif model_name == 'clip':
        model_path = kwargs.get('model_path', 'clip-vit-base-patch32')
        return CLIPViTClassifier(num_classes=num_classes, model_path=model_path)
    elif model_name == 'vit':
        model_path = kwargs.get('model_path', 'vit-base-patch16-224')
        return VitClassifier(num_classes=num_classes, pretrained_path=model_path)
    elif model_name == 'resnet34':
        pretrained = kwargs.get('pretrained', True)
        return ResNet34Classifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet152':
        pretrained = kwargs.get('pretrained', True)
        return ResNet152Classifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"未知的模型名称: {model_name}. 可选: 'dinov3', 'clip', 'resnet34'")


if __name__ == "__main__":
    # 测试模型
    print("=" * 50)
    print("测试 ResNet34")
    model = get_model('resnet34', num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    print("=" * 50)
    print("测试 ResNet101")
    model = get_model('resnet101', num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    print("\n" + "=" * 50)
    print("测试 CLIP ViT")
    try:
        model = get_model('clip', num_classes=6)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {out.shape}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"CLIP模型加载失败: {e}")
    
    print("\n" + "=" * 50)
    print("测试 vit")
    try:
        model = get_model('vit', num_classes=6)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {out.shape}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"DINOv3模型加载失败: {e}")

    print("\n" + "=" * 50)
    print("测试 DINOv3")
    try:
        model = get_model('dinov3', num_classes=6)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {out.shape}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    except Exception as e:
        print(f"DINOv3模型加载失败: {e}")

