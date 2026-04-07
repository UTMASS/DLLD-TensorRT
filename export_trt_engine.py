"""
Export ONNX model to TensorRT engine.

Usage:
  python export_trt_engine.py                                    # FP32
  python export_trt_engine.py --fp16                             # FP16 (recommended)
  python export_trt_engine.py --fp16 --onnx custom.onnx          # custom model
  python export_trt_engine.py --fp16 --output my_engine.trt      # custom output

Requirements:
  pip install tensorrt-cu12 torch
"""

import argparse
import os
import sys
import time

# Ensure CUDA DLLs from PyTorch are findable by TensorRT
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib) and _torch_lib not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path, fp16=False, max_workspace_gb=4):
    """Build a TensorRT engine from an ONNX model."""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX
    print(f"[TRT] Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [ERROR] {parser.get_error(i)}")
            return None

    # Print network info
    print(f"[TRT] Network inputs:")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  {inp.name}: {inp.shape} ({inp.dtype})")
    print(f"[TRT] Network outputs:")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  {out.name}: {out.shape} ({out.dtype})")

    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_gb * (1 << 30))

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 mode enabled")
        else:
            print("[TRT] WARNING: FP16 not supported on this GPU, falling back to FP32")

    # Build engine
    print("[TRT] Building engine (this may take a few minutes)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("[TRT] ERROR: Failed to build engine")
        return None

    elapsed = time.time() - t0
    size_mb = serialized_engine.nbytes / (1024 * 1024)
    print(f"[TRT] Engine built in {elapsed:.1f}s ({size_mb:.1f} MB)")
    return serialized_engine


def main():
    parser = argparse.ArgumentParser(description="Export ONNX to TensorRT engine")
    parser.add_argument("--onnx", type=str, default="culane_res18.onnx",
                        help="Path to ONNX model (default: culane_res18.onnx)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output engine path (default: <onnx_name>_fp16.trt or _fp32.trt)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision (recommended)")
    parser.add_argument("--workspace", type=int, default=4,
                        help="Max workspace size in GB (default: 4)")
    args = parser.parse_args()

    # Resolve ONNX path
    onnx_path = args.onnx
    if not os.path.isabs(onnx_path):
        onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), onnx_path)

    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(onnx_path)[0]
        suffix = "_fp16" if args.fp16 else "_fp32"
        output_path = f"{base}{suffix}.trt"

    # Build
    serialized = build_engine(onnx_path, fp16=args.fp16, max_workspace_gb=args.workspace)
    if serialized is None:
        sys.exit(1)

    # Save
    with open(output_path, "wb") as f:
        f.write(bytes(serialized))
    print(f"[TRT] Saved engine: {output_path}")
    print(f"[TRT] Use with: python lane_detection_dl.py --video input.mp4 --backend trt --trt-engine {output_path}")


if __name__ == "__main__":
    main()
