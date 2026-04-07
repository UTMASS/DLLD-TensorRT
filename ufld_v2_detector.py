"""
UFLD v2 Lane Detector — 4-lane detection with diamond-style rendering.
Wraps the CULane ResNet18 ONNX model for use in lane_detection.py.

All 4 lanes rendered with glowing diamond markers.
Only the ego lane (inner two lanes) gets filled.
"""

import cv2
import numpy as np
from collections import deque
from enum import Enum


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class CurvatureType(Enum):
    STRAIGHT = "Straight Ahead"
    EASY_LEFT = "Easy Left Curve"
    HARD_LEFT = "Hard Left Curve"
    EASY_RIGHT = "Easy Right Curve"
    HARD_RIGHT = "Hard Right Curve"
    UNKNOWN = "Calculating..."


class UFLDv2Detector:
    """
    Ultra-Fast Lane Detection v2 with Kalman-smoothed output.
    Detects 4 lanes: left-side, left-ego, right-ego, right-side.
    """

    # Lane colors (BGR): outer lanes dimmer, ego lanes brighter
    LANE_COLORS = [
        # (near_color, far_color) for each lane
        (np.array([200, 120, 0], dtype=np.float32), np.array([100, 40, 0], dtype=np.float32)),     # left-side (blue)
        (np.array([255, 255, 0], dtype=np.float32), np.array([200, 50, 0], dtype=np.float32)),     # left-ego (cyan)
        (np.array([255, 255, 0], dtype=np.float32), np.array([200, 50, 0], dtype=np.float32)),     # right-ego (cyan)
        (np.array([200, 120, 0], dtype=np.float32), np.array([100, 40, 0], dtype=np.float32)),     # right-side (blue)
    ]

    GLOW_COLORS = [
        (np.array([150, 80, 0], dtype=np.float32), np.array([80, 20, 0], dtype=np.float32)),       # left-side
        (np.array([255, 150, 0], dtype=np.float32), np.array([180, 0, 0], dtype=np.float32)),      # left-ego
        (np.array([255, 150, 0], dtype=np.float32), np.array([180, 0, 0], dtype=np.float32)),      # right-ego
        (np.array([150, 80, 0], dtype=np.float32), np.array([80, 20, 0], dtype=np.float32)),       # right-side
    ]

    # CULane config
    ROW_ANCHOR = np.linspace(0.42, 1, 72)
    COL_ANCHOR = np.linspace(0, 1, 81)
    CROP_RATIO = 0.6

    def __init__(self, model_path, skip_frames=3, ema_alpha=0.3, hood_crop=0.15,
                 car_img_path=None, backend="onnx", trt_engine=None):
        """
        Args:
            model_path: Path to UFLD v2 CULane ONNX model (used when backend='onnx').
            skip_frames: Run inference every N frames, interpolate between.
            ema_alpha: EMA smoothing factor (0=full smooth, 1=no smooth).
            hood_crop: Fraction of bottom to mask as hood (0=off).
            backend: 'onnx' or 'trt' — inference backend to use.
            trt_engine: Path to TensorRT engine file (required when backend='trt').
        """
        self._backend_type = backend

        if backend == "trt":
            from trt_backend import TRTBackendONNXCompat
            if trt_engine is None:
                raise ValueError("trt_engine path required when backend='trt'")
            self.session = TRTBackendONNXCompat(trt_engine)
        else:
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_h = input_shape[2]  # 320
        self.input_w = input_shape[3]  # 1600
        self.input_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32

        self.skip_frames = max(1, skip_frames)
        self.ema_alpha = ema_alpha
        self.hood_crop = hood_crop

        # Smoothed lane points (4 lanes, each a list of (x,y) tuples)
        self._smoothed_lanes = [[] for _ in range(4)]
        self._last_raw_lanes = [[] for _ in range(4)]
        self._lane_detected = [False] * 4
        self._frame_idx = 0

        # For HUD compatibility: store ego lane as left/right line endpoints
        self.last_left = None
        self.last_right = None
        self._last_left = None
        self._last_right = None

        # Load car icon
        self._car_img = None
        if car_img_path is None:
            import os
            car_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "car.png")
        if os.path.exists(car_img_path):
            self._car_img = cv2.imread(car_img_path, cv2.IMREAD_UNCHANGED)

        # ── Lane departure state (set externally by LDWS) ──
        # None = no departure, "LEFT" = drifting left, "RIGHT" = drifting right
        self._departure_side = None

        # ── Curvature estimation (LKAS) ──
        self._curvature_radius = float('inf')  # meters
        self._curvature_direction = "F"         # "L", "R", "F"
        self._curvature_type = CurvatureType.UNKNOWN
        self._curvature_history = deque(maxlen=10)
        self._direction_history = deque(maxlen=10)
        self._steering_angle = 0.0  # degrees, negative=left, positive=right
        # BEV perspective transform calibration (meters per pixel)
        self._ym_per_pix = 30.0 / 720
        self._xm_per_pix = 3.7 / 700
        # Thresholds
        self._hard_curve_threshold = 500    # meters — below = HARD curve
        self._straight_threshold = 2000     # meters — above = STRAIGHT

        active = self.session.get_providers()
        src = trt_engine if backend == "trt" else model_path
        print(f"[UFLDv2] Loaded: {src} (backend={backend})")
        print(f"[UFLDv2] Input: {self.input_w}x{self.input_h}, Providers: {active}")

    def _preprocess(self, image):
        """Prepare image for UFLD v2 inference."""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        new_h = int(self.input_h / self.CROP_RATIO)
        img = cv2.resize(img, (self.input_w, new_h)).astype(np.float32)
        img = img[-self.input_h:, :, :]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = ((img / 255.0 - mean) / std)
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
        return img.astype(self.input_dtype)

    def _postprocess(self, output, img_w, img_h, local_width=1):
        """Parse UFLD v2 output into 4 lane point lists."""
        out = {
            "loc_row": output[0], "loc_col": output[1],
            "exist_row": output[2], "exist_col": output[3]
        }

        num_grid_row = out['loc_row'].shape[1]
        num_cls_row = out['loc_row'].shape[2]
        num_grid_col = out['loc_col'].shape[1]
        num_cls_col = out['loc_col'].shape[2]

        max_indices_row = out['loc_row'].argmax(1)
        valid_row = out['exist_row'].argmax(1)
        max_indices_col = out['loc_col'].argmax(1)
        valid_col = out['exist_col'].argmax(1)

        row_lane_idx = [1, 2]  # left-ego, right-ego
        col_lane_idx = [0, 3]  # left-side, right-side

        lanes = {0: [], 1: [], 2: [], 3: []}  # left-side, left-ego, right-ego, right-side
        detected = [False] * 4

        # Row-anchor lanes (ego lanes)
        for i in row_lane_idx:
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = list(range(
                            max(0, max_indices_row[0, k, i] - local_width),
                            min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1
                        ))
                        out_tmp = (_softmax(out['loc_row'][0, all_ind, k, i]) *
                                   list(map(float, all_ind))).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * img_w
                        tmp.append((int(out_tmp), int(self.ROW_ANCHOR[k] * img_h)))

                lane_idx = 1 if i == 1 else 2
                lanes[lane_idx] = tmp
                if len(tmp) > 2:
                    detected[lane_idx] = True

        # Column-anchor lanes (side lanes)
        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = list(range(
                            max(0, max_indices_col[0, k, i] - local_width),
                            min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1
                        ))
                        out_tmp = (_softmax(out['loc_col'][0, all_ind, k, i]) *
                                   list(map(float, all_ind))).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * img_h
                        tmp.append((int(self.COL_ANCHOR[k] * img_w), int(out_tmp)))

                lane_idx = 0 if i == 0 else 3
                lanes[lane_idx] = tmp
                if len(tmp) > 2:
                    detected[lane_idx] = True

        return lanes, detected

    def _smooth_lanes(self, raw_lanes, detected):
        """EMA smoothing on lane points."""
        alpha = self.ema_alpha
        for i in range(4):
            if detected[i] and len(raw_lanes[i]) > 2:
                if len(self._smoothed_lanes[i]) == len(raw_lanes[i]):
                    # EMA blend
                    self._smoothed_lanes[i] = [
                        (int(alpha * r[0] + (1 - alpha) * s[0]),
                         int(alpha * r[1] + (1 - alpha) * s[1]))
                        for r, s in zip(raw_lanes[i], self._smoothed_lanes[i])
                    ]
                else:
                    self._smoothed_lanes[i] = list(raw_lanes[i])
                self._lane_detected[i] = True
            else:
                # Fade out: keep previous but mark undetected after several misses
                self._lane_detected[i] = len(self._smoothed_lanes[i]) > 0

    def _extract_ego_endpoints(self, h):
        """Extract left/right ego lane as (x1,y1,x2,y2) for HUD compatibility."""
        self.last_left = None
        self.last_right = None

        # Left ego (index 1)
        pts = self._smoothed_lanes[1]
        if len(pts) >= 2:
            # Bottom-most and top-most points
            sorted_pts = sorted(pts, key=lambda p: p[1], reverse=True)
            x1, y1 = sorted_pts[0]   # bottom
            x2, y2 = sorted_pts[-1]  # top
            self.last_left = (x1, y1, x2, y2)

        # Right ego (index 2)
        pts = self._smoothed_lanes[2]
        if len(pts) >= 2:
            sorted_pts = sorted(pts, key=lambda p: p[1], reverse=True)
            x1, y1 = sorted_pts[0]
            x2, y2 = sorted_pts[-1]
            self.last_right = (x1, y1, x2, y2)

    @staticmethod
    def _diamond(px, py, r):
        return np.array([[px, py - r], [px + r, py],
                         [px, py + r], [px - r, py]], dtype=np.int32)

    @staticmethod
    def _gradient_color(t, c_start, c_end):
        return (int(c_start[0] + t * (c_end[0] - c_start[0])),
                int(c_start[1] + t * (c_end[1] - c_start[1])),
                int(c_start[2] + t * (c_end[2] - c_start[2])))

    def _draw_lanes(self, result, h):
        """Draw all 4 lanes with diamond markers + ego lane fill."""
        # ── Departure-aware color overrides ──
        # When LDWS detects departure, the ego lane edge being crossed turns red
        departing = self._departure_side  # None, "LEFT", or "RIGHT"

        # Warning colors (BGR): red/orange for the departing side
        WARN_NEAR = np.array([0, 0, 255], dtype=np.float32)
        WARN_FAR = np.array([0, 0, 150], dtype=np.float32)
        def _get_lane_colors(lane_idx):
            """Return (color_near, color_far) for this lane, with departure override."""
            if departing and lane_idx in (1, 2):
                if (lane_idx == 1 and departing == "LEFT") or \
                   (lane_idx == 2 and departing == "RIGHT"):
                    return (WARN_NEAR, WARN_FAR)
            return self.LANE_COLORS[lane_idx]

        # ── Ego lane fill (only between left-ego and right-ego) ──
        left_pts = self._smoothed_lanes[1]
        right_pts = self._smoothed_lanes[2]
        if len(left_pts) > 2 and len(right_pts) > 2:
            fill_mask = np.zeros_like(result)
            left_sorted = sorted(left_pts, key=lambda p: p[1])
            right_sorted = sorted(right_pts, key=lambda p: p[1], reverse=True)
            poly = np.array(left_sorted + right_sorted, dtype=np.int32)

            num_strips = 20
            min_y = min(p[1] for p in left_sorted + right_sorted)
            max_y = max(p[1] for p in left_sorted + right_sorted)
            for s in range(num_strips):
                t0 = s / num_strips
                t1 = (s + 1) / num_strips
                strip_y0 = int(max_y - t0 * (max_y - min_y))
                strip_y1 = int(max_y - t1 * (max_y - min_y))
                alpha = 1.0 - t0
                if departing:
                    # Red-tinted fill when departing
                    b = int(80 * alpha)
                    g = int(40 * alpha)
                    r = int(255 * alpha)
                else:
                    b = int(255 * alpha)
                    g = int(180 * alpha)
                    r = int(120 * alpha)
                strip_mask = np.zeros(result.shape[:2], dtype=np.uint8)
                cv2.rectangle(strip_mask, (0, strip_y1), (result.shape[1], strip_y0), 255, -1)
                lane_mask = np.zeros(result.shape[:2], dtype=np.uint8)
                cv2.fillPoly(lane_mask, [poly], 255)
                combined_mask = cv2.bitwise_and(strip_mask, lane_mask)
                fill_mask[combined_mask > 0] = (b, g, r)

            result[:] = cv2.addWeighted(result, 0.8, fill_mask, 0.3, 0)

        # ── Diamond markers for all 4 lanes ──
        num_points = 11
        center_idx = num_points // 2
        min_radius = max(6, int(h * 0.008))
        max_radius = max(16, int(h * 0.025))

        outer_max_radius = max(10, int(h * 0.015))
        outer_min_radius = max(4, int(h * 0.005))

        # Solid diamond markers
        for lane_idx in range(4):
            pts = self._smoothed_lanes[lane_idx]
            if len(pts) < 3:
                continue

            is_ego = lane_idx in (1, 2)
            mr = max_radius if is_ego else outer_max_radius
            mnr = min_radius if is_ego else outer_min_radius
            color_near, color_far = _get_lane_colors(lane_idx)

            sorted_pts = sorted(pts, key=lambda p: p[1])
            step = max(1, len(sorted_pts) // num_points)
            sampled = sorted_pts[::step][:num_points]

            for i, (px, py) in enumerate(sampled):
                t = i / max(len(sampled) - 1, 1)
                dist = abs(i - len(sampled) // 2)
                radius = int(mr - (mr - mnr) * dist / max(len(sampled) // 2, 1))
                dc = self._gradient_color(t, color_near, color_far)
                cv2.fillPoly(result, [self._diamond(px, py, radius)], dc)

    def process_frame(self, frame, frame_idx):
        """
        Process a single frame. Returns (result_image, lane_count).
        Runs inference every skip_frames, smooths between.
        """
        h, w = frame.shape[:2]

        # Mask hood
        if self.hood_crop > 0:
            hood_y = int(h * (1 - self.hood_crop))
            frame_input = frame.copy()
            frame_input[hood_y:, :] = 0
        else:
            frame_input = frame

        # Run inference or reuse
        if frame_idx % self.skip_frames == 0 or frame_idx <= 1:
            blob = self._preprocess(frame_input)
            output = self.session.run(None, {self.input_name: blob})
            raw_lanes, detected = self._postprocess(output, w, h)
            self._last_raw_lanes = raw_lanes
            self._smooth_lanes(raw_lanes, detected)
        else:
            # Reuse smoothed lanes from last inference
            pass

        self._extract_ego_endpoints(h)
        self._estimate_curvature(w, h)

        # Count detected lanes
        lane_count = sum(1 for i in range(4) if self._lane_detected[i] and len(self._smoothed_lanes[i]) > 2)

        # Draw on frame
        result = frame.copy()
        self._draw_lanes(result, h)

        return result, lane_count

    def detect(self, frame):
        """
        Compatibility interface for lane_detection.py.
        Returns (result_image, mask, lane_count).
        """
        self._frame_idx += 1
        result, lane_count = self.process_frame(frame, self._frame_idx)
        mask = np.zeros_like(frame)
        # Store for HUD
        self._last_left = self.last_left
        self._last_right = self.last_right
        return result, mask, lane_count

    def _get_car_lateral_offset(self, frame_w):
        """
        Estimate car's lateral offset within the ego lane using detected ego lines.
        Returns value in [-1, 1]: -1 = touching left ego line, +1 = touching right, 0 = centered.
        """
        left = self.last_left   # (x1,y1,x2,y2) bottom-to-top
        right = self.last_right
        if left is None or right is None:
            return 0.0
        # Use bottom points (closest to car)
        left_x = left[0]
        right_x = right[0]
        lane_center = (left_x + right_x) / 2.0
        frame_center = frame_w / 2.0
        lane_width = max(right_x - left_x, 1)
        offset = (frame_center - lane_center) / (lane_width / 2.0)
        return max(-1.0, min(1.0, offset))

    def create_hud_panel(self, frame_h, frame_w, road_info=None):
        """Create HUD panel with dynamic lane count from UFLD detection and car positioning."""
        pw = int(frame_h * 0.35)
        ph = frame_h
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        panel[:] = (15, 15, 15)

        # Grid
        grid_spacing = int(pw * 0.08)
        for x in range(0, pw, grid_spacing):
            cv2.line(panel, (x, 0), (x, ph), (30, 30, 30), 1)
        for y in range(0, ph, grid_spacing):
            cv2.line(panel, (0, y), (pw, y), (30, 30, 30), 1)

        # Header
        header_h = int(ph * 0.06)
        cv2.rectangle(panel, (0, 0), (pw, header_h), (25, 25, 25), -1)
        cv2.line(panel, (0, header_h), (pw, header_h), (60, 60, 60), 1)
        title_scale = pw / 350
        cv2.putText(panel, "UFLD v2 HUD", (int(pw * 0.05), int(header_h * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 220, 220), 2)
        has_ego = self._lane_detected[1] or self._lane_detected[2]
        status_color = (0, 255, 0) if has_ego else (0, 0, 180)
        cv2.circle(panel, (pw - int(pw * 0.08), int(header_h * 0.5)), int(pw * 0.02), status_color, -1)

        # ── Determine lane structure from UFLD detection ──
        # Lane lines: 0=L-Side, 1=L-Ego, 2=R-Ego, 3=R-Side
        # Driving lanes between them:
        #   left lane  = between lines 0 and 1
        #   ego lane   = between lines 1 and 2 (always present if either ego line detected)
        #   right lane = between lines 2 and 3
        det = [self._lane_detected[i] and len(self._smoothed_lanes[i]) > 2 for i in range(4)]

        has_left_lane = det[0] and det[1]
        has_ego_lane = det[1] or det[2]
        has_right_lane = det[2] and det[3]

        # Build lane list: each entry is (lane_type, is_ego)
        # lane_type: "left", "ego", "right"
        lanes = []
        if has_left_lane:
            lanes.append("left")
        if has_ego_lane:
            lanes.append("ego")
        if has_right_lane:
            lanes.append("right")

        if not lanes:
            lanes = ["ego"]  # fallback: show at least ego lane

        num_lanes = len(lanes)
        ego_idx = lanes.index("ego") if "ego" in lanes else 0

        # Car lateral offset within ego lane
        car_offset = self._get_car_lateral_offset(frame_w)

        # ── BEV drawing ──
        bev_top = header_h + int(ph * 0.02)
        bev_bottom = int(ph * 0.85)
        bev_h = bev_bottom - bev_top
        bev_cx = pw // 2
        lane_w_px = min(int(pw * 0.22), int(pw * 0.7) // max(num_lanes, 1))
        road_w = num_lanes * lane_w_px
        num_strips = 20
        perspective_factor = 0.4

        # Road center offset so ego lane is centered
        ego_center_offset = (ego_idx - (num_lanes - 1) / 2.0) * lane_w_px
        road_cx = bev_cx - int(ego_center_offset)

        # Lane colors
        EGO_FILL = (60, 50, 30)
        ADJ_FILL = (30, 30, 25)
        EGO_LINE_COLOR = (255, 255, 0)      # cyan for ego boundaries
        SIDE_LINE_COLOR = (120, 120, 100)    # gray for side boundaries
        DASHED_COLOR = (80, 80, 60)          # dashed lane dividers

        # Curvature-based lateral shift
        curve_max_shift = self._steering_angle * pw * 0.012
        curve_max_shift = max(-pw * 0.35, min(pw * 0.35, curve_max_shift))

        def _cs(t):
            return int(curve_max_shift * t * t)

        for strip_i in range(num_strips):
            t = strip_i / num_strips
            t1 = (strip_i + 1) / num_strips
            y0 = int(bev_bottom - t * bev_h)
            y1 = int(bev_bottom - t1 * bev_h)
            narrow0 = 1.0 - t * perspective_factor
            narrow1 = 1.0 - t1 * perspective_factor
            alpha = 1.0 - t  # fade with distance
            cs0 = _cs(t)
            cs1 = _cs(t1)

            for li, lane_type in enumerate(lanes):
                is_ego = (lane_type == "ego")
                # Bottom edge of strip
                ll0 = int(road_cx + (li - num_lanes / 2.0) * lane_w_px * narrow0) + cs0
                lr0 = int(road_cx + (li + 1 - num_lanes / 2.0) * lane_w_px * narrow0) + cs0
                # Top edge of strip
                ll1 = int(road_cx + (li - num_lanes / 2.0) * lane_w_px * narrow1) + cs1
                lr1 = int(road_cx + (li + 1 - num_lanes / 2.0) * lane_w_px * narrow1) + cs1

                if is_ego:
                    fc = (int(EGO_FILL[0] * alpha), int(EGO_FILL[1] * alpha), int(EGO_FILL[2] * alpha))
                else:
                    fc = (int(ADJ_FILL[0] * alpha), int(ADJ_FILL[1] * alpha), int(ADJ_FILL[2] * alpha))
                strip_pts = np.array([
                    [ll0, y0], [ll1, y1], [lr1, y1], [lr0, y0]
                ], dtype=np.int32)
                cv2.fillPoly(panel, [strip_pts], fc)

            # Draw lane lines between/around lanes
            for li in range(num_lanes + 1):
                x0 = int(road_cx + (li - num_lanes / 2.0) * lane_w_px * narrow0) + cs0
                x1_l = int(road_cx + (li - num_lanes / 2.0) * lane_w_px * narrow1) + cs1
                is_outer = (li == 0 or li == num_lanes)
                is_ego_boundary = (li == ego_idx or li == ego_idx + 1)

                if is_outer:
                    cv2.line(panel, (x0, y0), (x1_l, y1), SIDE_LINE_COLOR, 2)
                elif is_ego_boundary:
                    cv2.line(panel, (x0, y0), (x1_l, y1), EGO_LINE_COLOR, 2)
                else:
                    if strip_i % 3 == 0:
                        cv2.line(panel, (x0, y0), (x1_l, y1), DASHED_COLOR, 1)

        # ── Car icon positioned in ego lane with lateral offset ──
        car_h = int(ph * 0.1)
        car_w = int(car_h * 0.6)
        car_top = bev_bottom - int(ph * 0.02) - car_h

        # Car X: centered in ego lane + offset
        ego_lane_cx = int(road_cx + (ego_idx + 0.5 - num_lanes / 2.0) * lane_w_px)
        car_shift = int(car_offset * lane_w_px * 0.3)
        car_cx = ego_lane_cx + car_shift
        car_left = car_cx - car_w // 2

        if self._car_img is not None:
            # Resize car PNG to fit
            img = self._car_img
            resized = cv2.resize(img, (car_w, car_h), interpolation=cv2.INTER_AREA)
            # Alpha blend onto panel
            y1, y2 = max(0, car_top), min(ph, car_top + car_h)
            x1, x2 = max(0, car_left), min(pw, car_left + car_w)
            ry1, ry2 = y1 - car_top, y2 - car_top
            rx1, rx2 = x1 - car_left, x2 - car_left
            if y2 > y1 and x2 > x1 and ry2 <= resized.shape[0] and rx2 <= resized.shape[1]:
                roi = resized[ry1:ry2, rx1:rx2]
                if roi.shape[2] == 4:
                    alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
                    bgr = roi[:, :, :3].astype(np.float32)
                    bg = panel[y1:y2, x1:x2].astype(np.float32)
                    panel[y1:y2, x1:x2] = (alpha * bgr + (1 - alpha) * bg).astype(np.uint8)
                else:
                    panel[y1:y2, x1:x2] = roi
        else:
            # Fallback rectangle
            cv2.rectangle(panel, (car_left, car_top), (car_left + car_w, car_top + car_h),
                           (200, 200, 200), -1)
            cv2.rectangle(panel, (car_left, car_top), (car_left + car_w, car_top + car_h),
                           (255, 255, 255), 1)

        # ── Road info from OSM ──
        info_scale = pw / 500
        if road_info:
            road_name = road_info.get("name", "")
            maxspeed = road_info.get("maxspeed", "")

            # Speed limit sign (top-right corner of BEV area)
            if maxspeed:
                speed_num = maxspeed.replace(" mph", "").replace(" km/h", "").strip()
                sign_r = int(pw * 0.07)
                sign_cx = pw - int(pw * 0.12)
                sign_cy = bev_top + sign_r + 5

                # White circle with red border
                cv2.circle(panel, (sign_cx, sign_cy), sign_r, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(panel, (sign_cx, sign_cy), sign_r, (0, 0, 200), 3, cv2.LINE_AA)
                # Inner thin red ring
                cv2.circle(panel, (sign_cx, sign_cy), sign_r - 3, (0, 0, 180), 1, cv2.LINE_AA)

                # Speed number
                spd_scale = sign_r / 22
                (tw, th), _ = cv2.getTextSize(speed_num, cv2.FONT_HERSHEY_SIMPLEX, spd_scale, 2)
                cv2.putText(panel, speed_num,
                            (sign_cx - tw // 2, sign_cy + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, spd_scale, (0, 0, 0), 2, cv2.LINE_AA)

                # Unit label below sign
                unit = "mph" if "mph" in maxspeed else "km/h"
                (uw, _), _ = cv2.getTextSize(unit, cv2.FONT_HERSHEY_SIMPLEX, info_scale * 0.6, 1)
                cv2.putText(panel, unit,
                            (sign_cx - uw // 2, sign_cy + sign_r + int(ph * 0.025)),
                            cv2.FONT_HERSHEY_SIMPLEX, info_scale * 0.6, (100, 100, 100), 1, cv2.LINE_AA)

            # Road name at bottom
            if road_name:
                cv2.putText(panel, road_name[:22], (int(pw * 0.05), int(ph * 0.97)),
                            cv2.FONT_HERSHEY_SIMPLEX, title_scale * 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        return panel

    def _estimate_curvature(self, img_w, img_h):
        """
        Estimate lane curvature using polynomial fitting on BEV-transformed points.
        Uses the same approach as Vehicle-CV-ADAS perspectiveTransformation.py.
        """
        left_pts = self._smoothed_lanes[1]   # left-ego
        right_pts = self._smoothed_lanes[2]  # right-ego

        if len(left_pts) < 5 and len(right_pts) < 5:
            self._curvature_type = CurvatureType.UNKNOWN
            return

        # ── Build BEV-like coordinates ──
        # Transform perspective lane points to a pseudo-BEV space
        # by inverting the perspective foreshortening
        def to_bev(pts, w, h):
            """Convert perspective points to pseudo-BEV coordinates."""
            bev = []
            for px, py in pts:
                # Depth factor: how far up the image (0=bottom, 1=top of lane region)
                depth = max(0.01, (h - py) / (h * 0.58))
                # Undo perspective: x spread increases with depth
                bev_x = (px - w / 2) / depth
                bev_y = h - py  # flip so bottom=0
                bev.append((bev_x, bev_y))
            return bev

        left_radius = float('inf')
        right_radius = float('inf')
        direction = "F"

        # Fit polynomial on left ego lane
        left_coeff = None
        if len(left_pts) >= 5:
            bev_left = to_bev(left_pts, img_w, img_h)
            ly = np.array([p[1] for p in bev_left])
            lx = np.array([p[0] for p in bev_left])
            try:
                left_coeff = np.polyfit(ly * self._ym_per_pix, lx * self._xm_per_pix, 2)
                y_eval = np.max(ly) * self._ym_per_pix
                left_radius = abs(((1 + (2 * left_coeff[0] * y_eval + left_coeff[1]) ** 2) ** 1.5)
                                  / (2 * left_coeff[0] + 1e-9))
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Fit polynomial on right ego lane
        right_coeff = None
        if len(right_pts) >= 5:
            bev_right = to_bev(right_pts, img_w, img_h)
            ry = np.array([p[1] for p in bev_right])
            rx = np.array([p[0] for p in bev_right])
            try:
                right_coeff = np.polyfit(ry * self._ym_per_pix, rx * self._xm_per_pix, 2)
                y_eval = np.max(ry) * self._ym_per_pix
                right_radius = abs(((1 + (2 * right_coeff[0] * y_eval + right_coeff[1]) ** 2) ** 1.5)
                                   / (2 * right_coeff[0] + 1e-9))
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Average curvature radius
        radii = []
        if left_radius < 1e6:
            radii.append(left_radius)
        if right_radius < 1e6:
            radii.append(right_radius)

        if radii:
            radius = np.mean(radii)
        else:
            radius = float('inf')

        # Determine direction from polynomial coefficients
        coeffs = []
        if left_coeff is not None:
            coeffs.append(left_coeff[0])
        if right_coeff is not None:
            coeffs.append(right_coeff[0])

        if coeffs:
            avg_coeff = np.mean(coeffs)
            if avg_coeff < -0.00015:
                direction = "L"
            elif avg_coeff > 0.00015:
                direction = "R"
            else:
                direction = "F"

        # Temporal smoothing
        self._curvature_history.append(radius)
        self._direction_history.append(direction)
        smoothed_radius = float(np.median(list(self._curvature_history)))

        # Direction: use mode of recent history
        from collections import Counter
        dir_counts = Counter(self._direction_history)
        smoothed_dir = dir_counts.most_common(1)[0][0]

        self._curvature_radius = smoothed_radius
        self._curvature_direction = smoothed_dir

        # Classify curvature
        if smoothed_radius > self._straight_threshold:
            self._curvature_type = CurvatureType.STRAIGHT
        elif smoothed_dir == "L":
            if smoothed_radius <= self._hard_curve_threshold:
                self._curvature_type = CurvatureType.HARD_LEFT
            else:
                self._curvature_type = CurvatureType.EASY_LEFT
        elif smoothed_dir == "R":
            if smoothed_radius <= self._hard_curve_threshold:
                self._curvature_type = CurvatureType.HARD_RIGHT
            else:
                self._curvature_type = CurvatureType.EASY_RIGHT
        else:
            self._curvature_type = CurvatureType.STRAIGHT

        # Steering angle (simplified: angle ~ 1/radius, capped at ±45°)
        if smoothed_radius < 1e6:
            angle_mag = min(45.0, (1.0 / smoothed_radius) * 1500)
            if smoothed_dir == "L":
                self._steering_angle = -angle_mag
            elif smoothed_dir == "R":
                self._steering_angle = angle_mag
            else:
                self._steering_angle = 0.0
        else:
            self._steering_angle = 0.0

    def create_lkas_panel(self, frame_h, frame_w):
        """
        Create LKAS (Lane Keeping Assist) panel with:
        - Steering wheel arc visualization
        - Curvature radius display
        - Turn direction indicator
        - Road curvature classification
        Same dark HUD style as the BEV panel.
        """
        pw = int(frame_h * 0.35)
        ph = frame_h
        panel = np.zeros((ph, pw, 3), dtype=np.uint8)
        panel[:] = (15, 15, 15)

        # Grid background
        grid_spacing = int(pw * 0.08)
        for x in range(0, pw, grid_spacing):
            cv2.line(panel, (x, 0), (x, ph), (30, 30, 30), 1)
        for y in range(0, ph, grid_spacing):
            cv2.line(panel, (0, y), (pw, y), (30, 30, 30), 1)

        title_scale = pw / 350
        info_scale = pw / 500

        # ── Header ──
        header_h = int(ph * 0.06)
        cv2.rectangle(panel, (0, 0), (pw, header_h), (25, 25, 25), -1)
        cv2.line(panel, (0, header_h), (pw, header_h), (60, 60, 60), 1)
        cv2.putText(panel, "LKAS", (int(pw * 0.05), int(header_h * 0.7)),
                    cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 220, 220), 2)

        # Status indicator
        ct = self._curvature_type
        if ct == CurvatureType.STRAIGHT:
            status_color = (0, 255, 0)
        elif ct in (CurvatureType.EASY_LEFT, CurvatureType.EASY_RIGHT):
            status_color = (0, 180, 255)
        elif ct in (CurvatureType.HARD_LEFT, CurvatureType.HARD_RIGHT):
            status_color = (0, 0, 255)
        else:
            status_color = (180, 180, 0)
        cv2.circle(panel, (pw - int(pw * 0.08), int(header_h * 0.5)),
                   int(pw * 0.02), status_color, -1)

        cx = pw // 2

        # ── Steering Arc Visualization ──
        arc_cy = header_h + int(ph * 0.22)
        arc_r = int(pw * 0.28)
        angle = self._steering_angle

        # Outer ring (dark)
        cv2.ellipse(panel, (cx, arc_cy), (arc_r + 8, arc_r + 8),
                    0, 180, 360, (35, 35, 35), -1)
        cv2.ellipse(panel, (cx, arc_cy), (arc_r + 8, arc_r + 8),
                    0, 180, 360, (50, 50, 50), 2)

        # Background arc (dimmed semicircle at top)
        cv2.ellipse(panel, (cx, arc_cy), (arc_r, arc_r),
                    0, 180, 360, (30, 30, 30), -1)

        # Active steering arc — color-coded
        if ct == CurvatureType.STRAIGHT:
            arc_color = (0, 200, 0)
            arc_glow = (0, 100, 0)
        elif ct in (CurvatureType.EASY_LEFT, CurvatureType.EASY_RIGHT):
            arc_color = (0, 200, 255)
            arc_glow = (0, 100, 128)
        elif ct in (CurvatureType.HARD_LEFT, CurvatureType.HARD_RIGHT):
            arc_color = (0, 80, 255)
            arc_glow = (0, 40, 128)
        else:
            arc_color = (128, 128, 0)
            arc_glow = (64, 64, 0)

        # Draw the steering arc: center is 270° (top), left is <270, right is >270
        arc_span = min(abs(angle) * 2.5, 80)
        if angle < -0.5:
            # Turning left
            start_angle = 270 - arc_span
            end_angle = 270
        elif angle > 0.5:
            # Turning right
            start_angle = 270
            end_angle = 270 + arc_span
        else:
            # Straight — small indicator at top
            start_angle = 265
            end_angle = 275

        # Glow layer
        glow_panel = panel.copy()
        cv2.ellipse(glow_panel, (cx, arc_cy), (arc_r + 4, arc_r + 4),
                    0, start_angle, end_angle, arc_glow, int(pw * 0.06))
        glow_panel = cv2.GaussianBlur(glow_panel, (15, 15), 0)
        cv2.addWeighted(glow_panel, 0.5, panel, 0.5, 0, panel)

        # Solid arc
        cv2.ellipse(panel, (cx, arc_cy), (arc_r, arc_r),
                    0, start_angle, end_angle, arc_color, int(pw * 0.04))

        # Center dot
        cv2.circle(panel, (cx, arc_cy), int(pw * 0.03), (60, 60, 60), -1)
        cv2.circle(panel, (cx, arc_cy), int(pw * 0.03), (100, 100, 100), 1)

        # Needle showing steering angle
        needle_len = arc_r - int(pw * 0.04)
        needle_angle_rad = np.radians(270 + angle * 2)
        nx = int(cx + needle_len * np.cos(needle_angle_rad))
        ny = int(arc_cy + needle_len * np.sin(needle_angle_rad))
        cv2.line(panel, (cx, arc_cy), (nx, ny), arc_color, 2, cv2.LINE_AA)
        cv2.circle(panel, (nx, ny), 4, arc_color, -1)

        # Angle text below arc
        angle_text = f"{abs(angle):.1f}" + chr(176)
        if angle < -0.5:
            angle_text += " L"
        elif angle > 0.5:
            angle_text += " R"
        (tw, th), _ = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, title_scale * 0.8, 2)
        cv2.putText(panel, angle_text, (cx - tw // 2, arc_cy + arc_r + int(ph * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, title_scale * 0.8, arc_color, 2)

        # ── Turn Direction Icon (drawn, not PNG) ──
        icon_cy = arc_cy + arc_r + int(ph * 0.14)
        icon_r = int(pw * 0.12)

        if ct in (CurvatureType.HARD_LEFT, CurvatureType.EASY_LEFT):
            # Left arrow
            arrow_color = (0, 200, 255) if ct == CurvatureType.EASY_LEFT else (0, 80, 255)
            # Arrow shaft
            cv2.line(panel, (cx + icon_r, icon_cy), (cx - icon_r // 2, icon_cy), arrow_color, 3, cv2.LINE_AA)
            # Arrow head
            head = np.array([
                [cx - icon_r, icon_cy],
                [cx - icon_r // 2, icon_cy - icon_r // 2],
                [cx - icon_r // 2, icon_cy + icon_r // 2]
            ], dtype=np.int32)
            cv2.fillPoly(panel, [head], arrow_color)
            # Curve indication
            cv2.ellipse(panel, (cx - icon_r // 3, icon_cy - icon_r),
                        (icon_r, icon_r), 0, 60, 120, arrow_color, 2, cv2.LINE_AA)

        elif ct in (CurvatureType.HARD_RIGHT, CurvatureType.EASY_RIGHT):
            # Right arrow
            arrow_color = (0, 200, 255) if ct == CurvatureType.EASY_RIGHT else (0, 80, 255)
            cv2.line(panel, (cx - icon_r, icon_cy), (cx + icon_r // 2, icon_cy), arrow_color, 3, cv2.LINE_AA)
            head = np.array([
                [cx + icon_r, icon_cy],
                [cx + icon_r // 2, icon_cy - icon_r // 2],
                [cx + icon_r // 2, icon_cy + icon_r // 2]
            ], dtype=np.int32)
            cv2.fillPoly(panel, [head], arrow_color)
            cv2.ellipse(panel, (cx + icon_r // 3, icon_cy - icon_r),
                        (icon_r, icon_r), 0, 60, 120, arrow_color, 2, cv2.LINE_AA)

        elif ct == CurvatureType.STRAIGHT:
            # Straight up arrow
            arrow_color = (0, 200, 0)
            cv2.line(panel, (cx, icon_cy + icon_r), (cx, icon_cy - icon_r // 2), arrow_color, 3, cv2.LINE_AA)
            head = np.array([
                [cx, icon_cy - icon_r],
                [cx - icon_r // 2, icon_cy - icon_r // 2],
                [cx + icon_r // 2, icon_cy - icon_r // 2]
            ], dtype=np.int32)
            cv2.fillPoly(panel, [head], arrow_color)
        else:
            # Unknown — question mark
            cv2.putText(panel, "?", (cx - int(icon_r * 0.4), icon_cy + int(icon_r * 0.4)),
                        cv2.FONT_HERSHEY_SIMPLEX, title_scale * 1.5, (128, 128, 0), 2)

        # Direction label
        dir_label = ct.value
        (tw, th), _ = cv2.getTextSize(dir_label, cv2.FONT_HERSHEY_SIMPLEX, info_scale * 0.9, 1)
        cv2.putText(panel, dir_label, (cx - tw // 2, icon_cy + icon_r + int(ph * 0.04)),
                    cv2.FONT_HERSHEY_SIMPLEX, info_scale * 0.9, arc_color, 1, cv2.LINE_AA)

        # ── Curvature Radius Display ──
        radius_y = icon_cy + icon_r + int(ph * 0.10)
        cv2.line(panel, (int(pw * 0.1), radius_y - int(ph * 0.02)),
                 (int(pw * 0.9), radius_y - int(ph * 0.02)), (40, 40, 40), 1)

        cv2.putText(panel, "CURVATURE RADIUS", (int(pw * 0.08), radius_y),
                    cv2.FONT_HERSHEY_SIMPLEX, info_scale * 0.7, (100, 100, 100), 1)

        if self._curvature_radius < 1e6:
            if self._curvature_radius < 1000:
                r_text = f"{self._curvature_radius:.0f} m"
            else:
                r_text = f"{self._curvature_radius / 1000:.1f} km"
        else:
            r_text = "--- (straight)"
        (tw, th), _ = cv2.getTextSize(r_text, cv2.FONT_HERSHEY_SIMPLEX, title_scale * 0.9, 2)
        cv2.putText(panel, r_text, (cx - tw // 2, radius_y + int(ph * 0.05)),
                    cv2.FONT_HERSHEY_SIMPLEX, title_scale * 0.9, (0, 220, 220), 2)

        # ── Road Curvature Visualization (mini preview) ──
        preview_top = radius_y + int(ph * 0.09)
        preview_h = int(ph * 0.20)
        preview_bottom = preview_top + preview_h
        preview_cx = cx

        # Draw a curved road preview based on curvature
        road_w = int(pw * 0.25)
        num_strips = 16

        for s in range(num_strips):
            t0 = s / num_strips
            t1 = (s + 1) / num_strips
            y0 = int(preview_bottom - t0 * preview_h)
            y1 = int(preview_bottom - t1 * preview_h)

            # Lateral offset based on curvature
            curve_shift_0 = int(self._steering_angle * 1.5 * t0 * t0)
            curve_shift_1 = int(self._steering_angle * 1.5 * t1 * t1)

            # Perspective narrowing
            narrow_0 = 1.0 - t0 * 0.5
            narrow_1 = 1.0 - t1 * 0.5
            hw0 = int(road_w * narrow_0 / 2)
            hw1 = int(road_w * narrow_1 / 2)

            alpha = max(0.2, 1.0 - t0 * 0.8)

            # Road surface
            strip = np.array([
                [preview_cx + curve_shift_0 - hw0, y0],
                [preview_cx + curve_shift_1 - hw1, y1],
                [preview_cx + curve_shift_1 + hw1, y1],
                [preview_cx + curve_shift_0 + hw0, y0]
            ], dtype=np.int32)
            fill = (int(40 * alpha), int(40 * alpha), int(30 * alpha))
            cv2.fillPoly(panel, [strip], fill)

            # Lane edges
            edge_color = (int(200 * alpha), int(200 * alpha), 0)
            cv2.line(panel, (preview_cx + curve_shift_0 - hw0, y0),
                     (preview_cx + curve_shift_1 - hw1, y1), edge_color, 1, cv2.LINE_AA)
            cv2.line(panel, (preview_cx + curve_shift_0 + hw0, y0),
                     (preview_cx + curve_shift_1 + hw1, y1), edge_color, 1, cv2.LINE_AA)

            # Center dashed line
            if s % 3 == 0:
                center_color = (int(80 * alpha), int(80 * alpha), int(60 * alpha))
                cv2.line(panel, (preview_cx + curve_shift_0, y0),
                         (preview_cx + curve_shift_1, y1), center_color, 1, cv2.LINE_AA)

        # Car dot at bottom of preview
        cv2.circle(panel, (preview_cx, preview_bottom - 5), 5, (0, 220, 220), -1)

        # ── Footer legend ──
        footer_y = int(ph * 0.92)
        cv2.line(panel, (0, footer_y), (pw, footer_y), (40, 40, 40), 1)
        label_scale = pw / 650
        label_x = int(pw * 0.06)
        row_h = int(ph * 0.025)

        ly = footer_y + row_h
        cv2.circle(panel, (label_x + 6, ly - 3), 5, (0, 200, 0), -1)
        cv2.putText(panel, "Straight", (label_x + int(pw * 0.09), ly),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (180, 180, 180), 1)

        ly2 = footer_y + row_h * 2
        cv2.circle(panel, (label_x + 6, ly2 - 3), 5, (0, 200, 255), -1)
        cv2.putText(panel, "Easy Curve", (label_x + int(pw * 0.09), ly2),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (180, 180, 180), 1)

        ly3 = footer_y + row_h * 3
        cv2.circle(panel, (label_x + 6, ly3 - 3), 5, (0, 80, 255), -1)
        cv2.putText(panel, "Hard Curve", (label_x + int(pw * 0.09), ly3),
                    cv2.FONT_HERSHEY_SIMPLEX, label_scale, (180, 180, 180), 1)

        # Outer border
        cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), (50, 50, 50), 2)

        return panel

    def close(self):
        if self._backend_type == "trt" and hasattr(self.session, 'close'):
            self.session.close()
