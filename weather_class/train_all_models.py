"""
训练所有三个模型的脚本
"""
import os
import subprocess
import time

def train_model(model_name, epochs=30, batch_size=32, lr=None):
    """训练单个模型"""
    print("\n" + "="*70)
    print(f"开始训练模型: {model_name}")
    print("="*70 + "\n")
    
    # 设置默认学习率
    if lr is None:
        lr = 0.001 if model_name == 'resnet34' else 0.0001
    
    cmd = [
        'python', 'train.py',
        '--model', model_name,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--lr', str(lr)
    ]
    
    start_time = time.time()
    
    try:
        subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"\n✓ {model_name} 训练完成! 用时: {elapsed_time/60:.2f} 分钟\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model_name} 训练失败: {e}\n")
        return False

def main():
    print("="*70)
    print("天气分类 - 批量训练所有模型")
    print("="*70)
    print("\n将依次训练以下模型:")
    print("  1. ResNet34")
    print("  2. CLIP ViT")
    print("  3. DINOv3")
    print()
    
    input("按回车键开始训练...")
    
    total_start = time.time()
    results = {}
    
    # 训练配置
    models_config = [
        {'name': 'resnet34', 'epochs': 200, 'batch_size': 32, 'lr': 0.0001},
        {'name': 'clip', 'epochs': 200, 'batch_size': 32, 'lr': 0.0001},
        {'name': 'dinov3', 'epochs': 200, 'batch_size': 16, 'lr': 0.0001},
        {'name': 'vit', 'epochs': 200, 'batch_size': 16, 'lr': 0.0001},
        {'name': 'resnet152', 'epochs': 200, 'batch_size': 16, 'lr': 0.0001},
    ]
    
    # 训练所有模型

    for config in models_config:
        success = train_model(
            config['name'], 
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )
        results[config['name']] = success
    
    # 总结
    total_time = time.time() - total_start
    print("\n" + "="*70)
    print("所有训练任务完成！")
    print("="*70)
    print(f"\n总用时: {total_time/60:.2f} 分钟")
    print("\n训练结果:")
    for model_name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"  {model_name}: {status}")
    
    print("\n模型权重保存在: checkpoints/ 目录")
    print("训练曲线保存在: checkpoints/*_training_curves.png")
    print()

if __name__ == "__main__":
    main()

