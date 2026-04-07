"""
Deep Learning Lane Detection (DLLD)
====================================
Standalone DL-only lane detection using Ultra-Fast-Lane-Detection v2.

Features extracted from CV_OSM project:
  - UFLD v2 ONNX inference (4-lane detection)
  - Frame skip / temporal regularization (run inference every N frames)
  - EMA smoothing on lane points
  - Curvature estimation via BEV polynomial fitting
  - Glowing diamond lane rendering

Install:
  pip install opencv-python numpy onnxruntime

Usage (ONNX backend — default):
  python lane_detection_dl.py --video highway.mp4
  python lane_detection_dl.py --video 0                          # webcam
  python lane_detection_dl.py --video highway.mp4 -o result.mp4  # save output
  python lane_detection_dl.py --image road.jpg

Usage (TensorRT backend — faster):
  python export_trt_engine.py --fp16                             # build engine once
  python lane_detection_dl.py --video highway.mp4 --backend trt --trt-engine culane_res18_fp16.trt
"""

import argparse
import os
import sys
import time
from collections import deque

# Ensure CUDA DLLs from PyTorch are findable by onnxruntime / TensorRT
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib) and _torch_lib not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

import cv2
import numpy as np

from ufld_v2_detector import UFLDv2Detector


def _resize_for_display(frame, max_width):
    """Resize frame to fit display if it exceeds max_width."""
    if max_width and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        return cv2.resize(frame, (max_width, int(frame.shape[0] * scale)))
    return frame


def create_detector(args):
    backend = getattr(args, "backend", "onnx")
    trt_engine = getattr(args, "trt_engine", None)

    # Resolve ONNX model path (needed for onnx backend, optional info for trt)
    model_path = args.onnx_model
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

    if backend == "trt":
        if trt_engine is None:
            print("[ERROR] --trt-engine is required when --backend trt")
            print("\nBuild an engine first:")
            print("  python export_trt_engine.py --fp16")
            sys.exit(1)
        if not os.path.isabs(trt_engine):
            trt_engine = os.path.join(os.path.dirname(os.path.abspath(__file__)), trt_engine)
        if not os.path.exists(trt_engine):
            print(f"[ERROR] TensorRT engine not found: {trt_engine}")
            sys.exit(1)
    else:
        if not os.path.exists(model_path):
            print(f"[ERROR] ONNX model not found: {model_path}")
            print("\nDownload a CULane ResNet18 ONNX model from:")
            print("  https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2")
            sys.exit(1)

    return UFLDv2Detector(
        model_path,
        skip_frames=args.skip_frames,
        ema_alpha=args.ema_alpha,
        hood_crop=args.hood_crop,
        backend=backend,
        trt_engine=trt_engine,
    )


def process_image(args):
    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Cannot read image: {args.image}")
        return

    detector = create_detector(args)
    h, w = image.shape[:2]

    start = time.time()
    result, mask, lane_count = detector.detect(image)
    elapsed = time.time() - start
    print(f"Detected {lane_count} lane(s) in {elapsed:.3f}s")

    output_path = args.output or args.image.rsplit(".", 1)[0] + "_dl." + args.image.rsplit(".", 1)[1]
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")

    if not args.no_display:
        display = _resize_for_display(result, args.display_width)
        cv2.imshow("DL Lane Detection", display)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(args):
    source = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = create_detector(args)

    writer = None
    writer_initialized = False
    frame_idx = 0
    fps_list = deque(maxlen=30)

    print(f"Video: {w}x{h} @ {fps:.1f}fps | skip={args.skip_frames} ema={args.ema_alpha} | Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        result, _, lane_count = detector.detect(frame)
        frame_idx += 1

        elapsed = time.time() - start
        fps_list.append(1.0 / elapsed if elapsed > 0 else 0)
        avg_fps = np.mean(fps_list)

        # HUD text on video
        cv2.putText(result, f"FPS: {avg_fps:.1f} | Lanes: {lane_count}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if total > 0:
            pct = frame_idx / total * 100
            cv2.putText(result, f"{frame_idx}/{total} ({pct:.0f}%)",
                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Deferred writer initialization
        if args.output and not writer_initialized:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, fps,
                                     (result.shape[1], result.shape[0]))
            writer_initialized = True

        if writer:
            writer.write(result)

        if not args.no_display:
            display = _resize_for_display(result, args.display_width)
            cv2.imshow("DL Lane Detection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {args.output}")
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_idx} frames.")


def main():
    parser = argparse.ArgumentParser(
        description="Deep Learning Lane Detection (UFLD v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lane_detection_dl.py --video highway.mp4
  python lane_detection_dl.py --video 0                          # webcam
  python lane_detection_dl.py --video highway.mp4 -o result.mp4
  python lane_detection_dl.py --image road.jpg
  python lane_detection_dl.py --video road.mp4 --skip-frames 5 --ema-alpha 0.2

TensorRT (build engine first with: python export_trt_engine.py --fp16):
  python lane_detection_dl.py --video highway.mp4 --backend trt --trt-engine culane_res18_fp16.trt
        """,
    )
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--video", type=str, help="Path to video file or '0' for webcam")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    parser.add_argument("--onnx-model", type=str, default="culane_res18.onnx",
                        help="Path to UFLD v2 CULane ONNX model (default: culane_res18.onnx)")
    parser.add_argument("--no-display", action="store_true", help="Don't show GUI window")
    parser.add_argument("--display-width", type=int, default=960,
                        help="Max display window width (default: 960)")

    # Backend selection
    backend_group = parser.add_argument_group("Inference Backend")
    backend_group.add_argument("--backend", type=str, default="onnx", choices=["onnx", "trt"],
                               help="Inference backend: 'onnx' (default) or 'trt' (TensorRT)")
    backend_group.add_argument("--trt-engine", type=str, default=None,
                               help="Path to TensorRT engine file (required with --backend trt)")

    # DL-specific tuning
    dl_group = parser.add_argument_group("DL Inference Tuning")
    dl_group.add_argument("--skip-frames", type=int, default=3,
                          help="Run inference every N frames, reuse between (default: 3)")
    dl_group.add_argument("--ema-alpha", type=float, default=0.3,
                          help="EMA smoothing factor: 0=full smooth, 1=no smooth (default: 0.3)")
    dl_group.add_argument("--hood-crop", type=float, default=0.15,
                          help="Fraction of bottom to mask as hood, 0=off (default: 0.15)")

    args = parser.parse_args()

    if not args.image and not args.video:
        parser.print_help()
        return

    if args.image:
        process_image(args)
    elif args.video:
        process_video(args)


if __name__ == "__main__":
    main()
