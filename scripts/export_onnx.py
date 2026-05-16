from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(
    format='onnx',
    opset=12,        # TensorRT 8.6 推荐 12
    imgsz=640,
    dynamic=False,   # 先固定尺寸，简化 TensorRT 构建
    simplify=True
)
