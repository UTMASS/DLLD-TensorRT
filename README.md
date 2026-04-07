# Ultra-Fast-Lane-Detection V2 TensorRT

A self-contained, real-time lane detection system built on Ultra-Fast-Lane-Detection v2 (UFLD v2). Detects up to four lanes from dashcam video or webcam feeds using ONNX Runtime or TensorRT inference. This folder contains everything needed to run the system -- just add the model weights.

---

## What Is in This Folder

```
real-time/
  lane_detection_dl.py      Main entry point — video, image, and webcam processing
  ufld_v2_detector.py       Core detector — preprocessing, inference, postprocessing, rendering
  trt_backend.py            TensorRT backend — drop-in replacement for ONNX Runtime
  export_trt_engine.py      ONNX-to-TensorRT engine builder (FP16 / FP32)
  benchmark.py              Performance comparison between ONNX and TensorRT backends
  requirements.txt          Python dependencies
  LICENSE                   MIT license
  README.md                 This file
```

## Getting the Model Weights

Download the CULane ResNet18 or BDD100K ONNX model from the [Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) repository and place both files in this folder:

```
culane_res18.onnx          (the model header)
culane_res18.onnx.data     (the weight data, ~787 MB)
```

Alternatively, [PINTO0309/PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) provides pre-converted UFLD v2 models in multiple formats.

---

## Installation

```bash
pip install -r requirements.txt
```

This installs the base dependencies: OpenCV, NumPy, and ONNX Runtime.

For GPU-accelerated ONNX inference:

```bash
pip install onnxruntime-gpu
```

For TensorRT (see the TensorRT section below for details):

```bash
pip install tensorrt-cu12 torch
```

---

## Running the Lane Detector

### ONNX backend (default, works on CPU or GPU)

```bash
python lane_detection_dl.py --video dashcam.mp4
python lane_detection_dl.py --video 0                          # webcam
python lane_detection_dl.py --image road.jpg
python lane_detection_dl.py --video dashcam.mp4 -o result.mp4  # save annotated output
```

### TensorRT backend (faster, requires NVIDIA GPU)

First, build the TensorRT engine from the ONNX model (one-time step):

```bash
python export_trt_engine.py --fp16
```

Then run with the TensorRT backend:

```bash
python lane_detection_dl.py --video dashcam.mp4 --backend trt --trt-engine culane_res18_fp16.trt
```

### Tuning parameters

```bash
python lane_detection_dl.py --video dashcam.mp4 \
    --skip-frames 3      \   # run inference every N frames (default: 3)
    --ema-alpha 0.3      \   # smoothing factor, 0=smooth 1=raw (default: 0.3)
    --hood-crop 0.15     \   # mask bottom 15% as hood (default: 0.15)
    --no-display             # headless mode, no GUI window
```

### Benchmarking

```bash
python benchmark.py
```

Compares ONNX Runtime and TensorRT on 100 frames from a reference video.

---

## How It Works

The pipeline runs in five stages for each frame:

1. **Preprocessing** — Resize to 1600x320, crop, normalize with ImageNet statistics, convert to NCHW tensor.
2. **Inference** — Feed the tensor through the UFLD v2 model (ONNX or TensorRT). The model outputs four tensors encoding row-anchor and column-anchor predictions for four lanes.
3. **Postprocessing** — Decode anchor predictions into pixel coordinates using softmax-weighted refinement. Apply EMA temporal smoothing to reduce jitter between frames.
4. **Curvature estimation** — Transform ego-lane points into a pseudo bird's-eye-view, fit second-order polynomials, and compute curvature radius and steering angle.
5. **Rendering** — Draw diamond markers on all four lanes, fill the ego lane with a gradient overlay, and optionally display HUD panels.

Frame skipping (default: every 3rd frame) reduces GPU load by reusing the previous detection on intermediate frames. Combined with EMA smoothing, this yields stable output at 20+ FPS on a desktop GPU with TensorRT FP16.

---

## Suggestions for Better Real-Time Performance

The system already achieves strong frame rates. The ideas below are about going further -- tighter latency, lower power, and deterministic execution on embedded hardware.

### Where the time goes

| Stage | Approx. Time (GPU) | Notes |
|---|---|---|
| Preprocessing (resize, normalize) | 1-2 ms | CPU-bound, memory copies |
| Inference (TensorRT FP16) | 3-6 ms | GPU-bound, main bottleneck |
| Postprocessing (softmax, anchors) | 1-3 ms | CPU-bound (NumPy) |
| Rendering (overlays, HUD) | 2-4 ms | CPU-bound (OpenCV drawing) |
| Total per frame | ~8-15 ms | 20 FPS theoretical |

### INT8 quantization

We currently run at FP16 precision, which halves memory bandwidth compared to FP32 and roughly doubles throughput on Tensor Cores. Going one step further to INT8 cuts bandwidth in half again.

INT8 inference on TensorRT typically runs 2-4x faster than FP16 on the same GPU. On embedded boards like Jetson Orin, this can be the difference between hitting 30 FPS and missing it. The reduced memory footprint also matters when multiple models share the same GPU.

INT8 requires a calibration step. You feed a representative set of images (a few hundred frames from typical driving footage) through the network, and TensorRT measures activation ranges to choose per-tensor scale factors:

1. Collect 200-500 representative frames covering varied lighting, road types, and weather.
2. Write a calibration data loader that feeds these through the existing preprocessing pipeline.
3. Pass the calibrator to the TensorRT builder:

```python
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = MyCalibrator(calib_images, batch_size=8)
```

4. Validate the resulting engine against FP16 outputs to check for accuracy loss (usually under 1% for lane detection).

The main risk is that aggressive quantization can distort the softmax outputs used for sub-pixel lane refinement. If that happens, mixed precision (INT8 backbone, FP16 head) is a practical fallback.

Expected gains:
- Inference time: 1.5-3 ms per frame (down from 3-6 ms at FP16)
- Engine size: roughly 100 MB (down from 394 MB)
- Power consumption: meaningfully lower on thermally-constrained platforms

### GPU preprocessing and postprocessing

Right now, preprocessing runs on the CPU using OpenCV and NumPy. The result is copied to the GPU for inference, and outputs are copied back for postprocessing. These memory transfers add latency and create synchronization points.

OpenCV's `cv2.cuda` module can handle resize, color conversion, and basic arithmetic on the GPU. Alternatively, a small CUDA kernel or CuPy script can fuse all preprocessing into a single pass -- resize, crop, normalize, and transpose -- without leaving the GPU. This eliminates two host-device round trips per frame.

The softmax and argmax in postprocessing are natural fits for GPU execution too. The payoff is smaller (1-3 ms), but on embedded hardware every millisecond counts.

### Asynchronous pipeline (double buffering)

The current implementation processes frames sequentially: capture, preprocess, infer, postprocess, render, display. Each stage waits for the previous one.

A pipelined approach overlaps stages using multiple CUDA streams and CPU threads:

```
Time -->
Stream A: [preprocess N]  [infer N]       [postprocess N]
Stream B:                  [preprocess N+1] [infer N+1]     [postprocess N+1]
Display:                                    [render N-1]    [render N]
```

This reduces effective per-frame time to the duration of the slowest single stage rather than the sum. The trade-off is complexity -- buffer lifetimes, stream synchronization, and a one-frame display delay. The existing frame-skip mechanism helps here by giving the pipeline natural breathing room.

### Model architecture options

The current ResNet18 backbone is already lightweight, but there are further options:

- **MobileNet or EfficientNet backbones** trade a small amount of accuracy for significantly fewer FLOPs. Useful when sharing GPU resources with other perception models.
- **Knowledge distillation** trains a smaller student model to mimic the current outputs. Can yield a model half the size with minimal accuracy loss.
- **Pruning** removes redundant filters. TensorRT optimizes the sparser network more aggressively.

These compound with INT8 quantization -- a pruned MobileNet backbone at INT8 could run inference in under 1 ms on a Jetson Orin.

---

## Why C++ Matters for Production

Python is excellent for prototyping. This codebase proves it -- we went from concept to a working 20 FPS system quickly. But for production deployment in a vehicle, C++ offers advantages that are hard to replicate in Python.

### Predictable latency

Python's garbage collector introduces unpredictable pauses of several milliseconds. In a control loop at 30+ Hz, a 10 ms GC pause means a dropped frame. C++ gives you deterministic memory management -- you control exactly when allocations and deallocations happen.

### Lower per-frame overhead

The Python interpreter adds overhead to every function call, every loop iteration, every array operation. NumPy hides this for bulk math, but the glue code between operations runs at Python speed. In C++, this compiles to tight machine instructions.

A single Python function call costs roughly 50-100 nanoseconds. In C++ it is under 5. Negligible for one call, but thousands of calls per frame (loop iterations, small utilities) makes it add up.

### Direct hardware access

On embedded platforms (Jetson, custom ECUs), you need to interface with camera drivers, CAN bus, or display hardware through system-level APIs. These are native C/C++. Calling them from Python means going through bindings that add latency and complexity.

### Smaller deployment footprint

A compiled C++ binary with statically linked TensorRT weighs a few megabytes. A Python deployment needs the interpreter, NumPy, OpenCV, PyTorch (used here only for CUDA memory management), and all their transitive dependencies. On embedded targets with limited storage, this matters.

### What a C++ port would look like

TensorRT's native API is C++, so the inference backend actually becomes simpler without the Python wrapper. The main work would be:

1. Rewrite preprocessing using OpenCV's C++ API or raw CUDA kernels.
2. Port postprocessing (softmax, anchor decoding, EMA smoothing) with Eigen or plain loops.
3. Port rendering (OpenCV C++ drawing is nearly identical to Python).
4. Wire it together with a capture loop using VideoCapture or GStreamer.

A reasonable estimate is 800-1200 lines of C++ (compared to roughly 1500 lines of Python), because argument parsing and string formatting either go away or get simpler.

### A pragmatic middle ground

You do not have to rewrite everything at once. A common approach:

- Keep the Python code as the reference implementation and experimentation platform.
- Write the inference-critical path (preprocess, infer, postprocess) as a C++ shared library.
- Call it from Python using pybind11 or ctypes for validation and testing.
- For final deployment, build a standalone C++ binary.

This way you get production performance without losing the ability to quickly test new ideas in Python.

---

## Summary of Expected Gains

Starting from the current baseline (TensorRT FP16, skip=3, ~20 FPS on desktop GPU):

| Optimization | Estimated Speedup | Effort |
|---|---|---|
| INT8 quantization | 1.5-2x inference | Medium -- calibration dataset needed |
| GPU preprocessing | 10-20% overall | Low to medium |
| Async pipeline | 20-40% throughput | Medium |
| C++ rewrite | 30-50% overall | High -- but necessary for production |
| Lighter backbone | 2-3x inference | High -- retraining needed |

These are not mutually exclusive. Combining INT8, GPU preprocessing, and C++ could bring per-frame latency under 5 ms on desktop hardware and under 15 ms on embedded platforms -- well within the budget for real-time vehicle control.

---

## Next Steps

1. Build an INT8 calibration pipeline using representative driving footage.
2. Profile the current pipeline end-to-end with `nsys` or TensorRT's built-in profiler to find the actual bottlenecks.
3. Prototype GPU preprocessing with CuPy to validate the latency savings before committing to CUDA kernels.
4. Evaluate whether the deployment target justifies the C++ port now or later.

The goal is not to optimize everything at once, but to make informed choices about where each millisecond of improvement matters most for the target platform.
