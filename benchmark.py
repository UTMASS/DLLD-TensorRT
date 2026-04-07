"""Benchmark ONNX vs TensorRT backends."""
import os, sys, time

# CUDA DLL fix
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np
from ufld_v2_detector import UFLDv2Detector

VIDEO = r"C:\Users\ahmad\Downloads\vehicle_crop.mp4"
N_FRAMES = 100


def bench(name, detector):
    cap = cv2.VideoCapture(VIDEO)
    times = []
    for i in range(N_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        detector.detect(frame)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    cap.release()
    detector.close()

    # Skip first 5 warmup frames
    times = times[5:]
    avg_ms = np.mean(times) * 1000
    fps = 1000.0 / avg_ms
    print(f"  {name:20s} | {avg_ms:7.1f} ms | {fps:6.1f} FPS")


print(f"Benchmarking {N_FRAMES} frames (skip_frames=1, no rendering overhead)")
print("-" * 55)

# TensorRT FP16
trt_engine = "culane_res18_fp16.trt"
if os.path.exists(trt_engine):
    d = UFLDv2Detector("culane_res18.onnx", skip_frames=1, ema_alpha=1.0,
                       backend="trt", trt_engine=trt_engine)
    bench("TensorRT FP16", d)

# ONNX (GPU or CPU, whichever loads)
d = UFLDv2Detector("culane_res18.onnx", skip_frames=1, ema_alpha=1.0, backend="onnx")
bench("ONNX Runtime", d)

print("-" * 55)
