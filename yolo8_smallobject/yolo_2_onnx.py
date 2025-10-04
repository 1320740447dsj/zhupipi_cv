from ultralytics import YOLO

# 加载 YOLO 模型的 .pt 文件
model = YOLO(r"./yolo11m.pt")

print("正在导出 ONNX 模型到 GPU (device=0) 并启用 FP16 (half=True)...")

# 导出模型，启用 GPU 和 FP16
model.export(
    format="onnx",      # 导出格式为 ONNX
    imgsz=(640, 640),   # 输入图像的尺寸
    half=True,          # 【关键修改】启用 FP16 量化，需要 GPU 支持
    simplify=True,      # 简化 ONNX 模型
    batch=1,            # 批处理大小
    device="0"          # 【关键修改】指定导出设备为 GPU 0
)

print("ONNX 模型已成功导出为 yolo11m.onnx (或其他默认名称)。")