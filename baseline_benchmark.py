import time
import numpy as np
import cv2
import subprocess

# 读取图片
img_path = "/workspace/demo/demo.jpg"
img = cv2.imread(img_path)
if img is None:
    print(f"ERROR: Cannot load image from {img_path}")
    exit(1)

# ---------- Ultralytics 原生推理基线 ----------
from ultralytics import YOLO
model = YOLO("/workspace/demo/yolov8n.pt")

latencies = []
print("Warmup Ultralytics...")
for _ in range(5):
    model.predict(img, imgsz=640, verbose=False)
print("Testing Ultralytics (100 runs)...")
for _ in range(100):
    t0 = time.perf_counter()
    model.predict(img, imgsz=640, verbose=False)
    latencies.append((time.perf_counter() - t0) * 1000)
avg_ultra = np.mean(latencies)
std_ultra = np.std(latencies)
print(f"Ultralytics: avg={avg_ultra:.2f}ms, std={std_ultra:.2f}ms")

# ---------- ONNX Runtime 基线 ----------
import onnxruntime as ort
sess = ort.InferenceSession("/workspace/demo/yolov8n.onnx",
                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name

def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0
    return np.ascontiguousarray(img)

input_data = preprocess(img)
latencies = []
print("Warmup ONNX Runtime...")
for _ in range(5):
    sess.run(None, {input_name: input_data})
print("Testing ONNX Runtime (100 runs)...")
for _ in range(100):
    t0 = time.perf_counter()
    sess.run(None, {input_name: input_data})
    latencies.append((time.perf_counter() - t0) * 1000)
avg_ort = np.mean(latencies)
std_ort = np.std(latencies)
print(f"ONNX Runtime GPU: avg={avg_ort:.2f}ms, std={std_ort:.2f}ms")

# ---------- 资源占用 ----------
try:
    mem_used = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']).decode().strip()
except:
    mem_used = "N/A"
print(f"GPU memory used: {mem_used} MiB")

# 保存结果
with open("/workspace/demo/baseline_results.txt", "w") as f:
    f.write(f"Ultralytics: avg={avg_ultra:.2f}ms, std={std_ultra:.2f}ms\n")
    f.write(f"ONNX Runtime GPU: avg={avg_ort:.2f}ms, std={std_ort:.2f}ms\n")
    f.write(f"GPU memory used: {mem_used} MiB\n")
print("Baseline results saved.")
