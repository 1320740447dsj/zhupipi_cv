import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models import get_model

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class WeatherPredictor:
    """天气分类预测器"""
    
    def __init__(self, model_name, checkpoint_path, device='cuda'):
        """
        初始化预测器
        
        Args:
            model_name: 模型名称 ('dinov3', 'clip', 'resnet34')
            checkpoint_path: 模型权重路径
            device: 设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载checkpoint
        print(f"加载模型权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取类别信息
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        else:
            # 默认类别
            self.class_names = ['sunny', 'snow', 'rain', 'lightning', 'fogsmog', 'cloudy']
        
        self.num_classes = len(self.class_names)
        print(f"类别数: {self.num_classes}, 类别: {self.class_names}")
        
        # 创建模型
        print(f"创建模型: {model_name}")
        self.model = get_model(model_name, num_classes=self.num_classes)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("模型加载完成！\n")
    
    def predict(self, image_path, top_k=3):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            top_k: 返回前k个预测结果
        
        Returns:
            predictions: 预测结果列表 [(类别, 概率), ...]
        """
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        # 获取top-k结果
        probs, indices = torch.topk(probabilities, top_k)
        probs = probs.cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]
        
        # 构建结果
        predictions = [(self.class_names[idx], prob) for idx, prob in zip(indices, probs)]
        
        return predictions, image
    
    def predict_and_visualize(self, image_path, save_path=None):
        """
        预测并可视化结果
        
        Args:
            image_path: 图片路径
            save_path: 保存路径（可选）
        """
        # 预测
        predictions, image = self.predict(image_path)
        
        # 打印结果
        print(f"图片: {image_path}")
        print("预测结果:")
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"  {i}. {class_name}: {prob * 100:.2f}%")
        print()
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 显示图片
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f'输入图片\n预测: {predictions[0][0]} ({predictions[0][1]*100:.1f}%)', 
                     fontsize=14, fontweight='bold')
        
        # 显示概率分布
        classes = [pred[0] for pred in predictions]
        probs = [pred[1] for pred in predictions]
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(classes))]
        
        bars = ax2.barh(classes, probs, color=colors)
        ax2.set_xlabel('概率', fontsize=12)
        ax2.set_title('Top-3 预测结果', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        
        # 在柱状图上显示数值
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{prob*100:.1f}%', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description='天气分类模型预测')
    
    parser.add_argument('--model', type=str, required=True, 
                       choices=['dinov3', 'clip', 'resnet34'], help='模型名称')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重路径')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--save', type=str, default=None, help='结果保存路径（可选）')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='设备')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型权重文件不存在: {args.checkpoint}")
        return
    
    if not os.path.exists(args.image):
        print(f"错误: 图片文件不存在: {args.image}")
        return
    
    # 创建预测器
    predictor = WeatherPredictor(args.model, args.checkpoint, args.device)
    
    # 预测并可视化
    predictor.predict_and_visualize(args.image, args.save)


if __name__ == "__main__":
    # 如果直接运行，使用交互式预测
    import sys
    
    if len(sys.argv) == 1:
        print("=" * 60)
        print("天气分类预测系统 - 交互模式")
        print("=" * 60)
        
        # 选择模型
        print("\n可用模型:")
        print("1. resnet34")
        print("2. clip")
        print("3. dinov3")
        model_choice = input("选择模型 (1/2/3): ").strip()
        model_map = {'1': 'resnet34', '2': 'clip', '3': 'dinov3'}
        model_name = model_map.get(model_choice, 'resnet34')
        
        # 输入权重路径
        checkpoint_path = input(f"输入{model_name}模型权重路径 (默认: checkpoints/{model_name}_best.pth): ").strip()
        if not checkpoint_path:
            checkpoint_path = f'checkpoints/{model_name}_best.pth'
        
        # 输入图片路径
        image_path = input("输入图片路径: ").strip()
        
        # 是否保存结果
        save_path = input("输入结果保存路径 (直接回车跳过): ").strip()
        if not save_path:
            save_path = None
        
        # 检查文件
        if not os.path.exists(checkpoint_path):
            print(f"\n错误: 模型权重文件不存在: {checkpoint_path}")
            sys.exit(1)
        
        if not os.path.exists(image_path):
            print(f"\n错误: 图片文件不存在: {image_path}")
            sys.exit(1)
        
        # 预测
        print("\n" + "=" * 60)
        predictor = WeatherPredictor(model_name, checkpoint_path)
        predictor.predict_and_visualize(image_path, save_path)
    else:
        # 命令行模式
        main()

