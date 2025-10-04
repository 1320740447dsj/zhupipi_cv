import cv2
from ultralytics import YOLO
import time

# --- 配置部分 ---
MODEL_PATH = 'yolo11m.pt'
CAMERA_INDEX = 0


def run_yolo_detection():
    # 1. 加载模型
    try:
        model = YOLO(MODEL_PATH)
        print(f"成功加载模型: {MODEL_PATH}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请检查 MODEL_PATH 是否正确，并确保模型文件存在。")
        return

    # 2. 打开摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"无法打开摄像头 (Index: {CAMERA_INDEX})。请检查摄像头是否连接或索引是否正确。")
        return

    # 3. 初始化 FPS 变量
    prev_time = time.time()

    # 4. 实时检测循环
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("无法从摄像头接收帧，退出...")
                break

            # 运行 YOLO 检测
            # stream=True 启用生成器模式，可以稍微提高效率
            results = model.predict(source=frame, stream=True, verbose=False)

            # 遍历检测结果并绘制
            for result in results:
                # 将结果中的边界框绘制到帧上
                annotated_frame = result.plot()

                # --- FPS 计算部分 ---
            current_time = time.time()
            # FPS = 1 / (当前时间 - 上一帧时间)
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # 在帧上显示 FPS
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(annotated_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # 显示结果帧
            cv2.imshow("YOLO Live Detection", annotated_frame)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # 5. 清理资源
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出，资源已释放。")


if __name__ == "__main__":
    run_yolo_detection()