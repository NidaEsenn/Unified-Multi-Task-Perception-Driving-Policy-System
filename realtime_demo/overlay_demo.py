"""Realtime demo that overlays perception outputs on video frames.

This demo loads simple detection and segmentation components and overlays
their outputs on a camera or video stream for quick visualization.
"""
from __future__ import annotations

from typing import Any
from collections import deque
from pathlib import Path
import math
import argparse
import logging

import cv2
import numpy as np

from perception.vehicle_detection_yolov8 import VehicleDetector, draw_detections
from perception.lane_unet import LaneSegmentationModel
from perception.drivable_area_unet import DrivableAreaSegmentationModel
import torch
from typing import Optional
from utils.config import CONFIG, POLICY_CHECKPOINT_DIR

logger = logging.getLogger(__name__)


def overlay_masks(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 0, 255), alpha: float = 0.4) -> np.ndarray:
    """Overlay a single-channel mask on `frame` using `color` and `alpha` blending."""
    vis = frame.copy()
    mask_rgb = np.zeros_like(frame, dtype=np.uint8)
    mask_rgb[..., 0] = (mask * 255).astype(np.uint8)
    blended = cv2.addWeighted(vis, 1.0, mask_rgb, alpha, 0)
    return blended


def predict_lane(model: LaneSegmentationModel, frame: np.ndarray) -> np.ndarray:
    """Return float mask HxW in [0,1] from lane model."""
    mask, overlay = model.infer_on_image(frame)
    return mask


def predict_drivable(model: DrivableAreaSegmentationModel, frame: np.ndarray) -> np.ndarray:
    """Return float mask HxW in [0,1] from drivable model."""
    mask, overlay = model.infer_on_image(frame)
    return mask


def run_demo(
    input_video: str | Path | int,
    output_video: str | Path | None = None,
    seq_len: int = 3,
    device: Optional[str] = None,
    weights_det: Optional[str] = None,
    weights_lane: Optional[str] = None,
    weights_drivable: Optional[str] = None,
    weights_policy: Optional[str] = None,
    debug_det: bool = False,
    debug_policy: bool = False,
    max_frames: Optional[int] = None,
    ) -> None:
    """Run the realtime overlay demo and write annotated video.

    Args:
        input_video: camera index (int) or input path
        output_video: Optional output path for video; if None only shows GUI
        seq_len: sequence length for ConvLSTM policy
        device: device string to use (mps/cuda/cpu)
        weights_det: detector weights path
        weights_lane: lane model weights
        weights_drivable: drivable model weights
        weights_policy: policy model weights
    """
    src = int(input_video) if isinstance(input_video, str) and str(input_video).isdigit() else input_video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # device selection
    device = device if device else CONFIG.DEVICE
    logger.info("Using device: %s", device)

    det_weights = weights_det if weights_det else f"{CONFIG.CHECKPOINT_DIR}/yolov8n.pt"
    lane_weights = weights_lane if weights_lane else f"{CONFIG.CHECKPOINT_DIR}/lane_unet.pth"
    drivable_weights = weights_drivable if weights_drivable else f"{CONFIG.CHECKPOINT_DIR}/drivable_unet.pth"
    policy_weights = weights_policy if weights_policy else f"{POLICY_CHECKPOINT_DIR}/policy_best.pt"

    detector = VehicleDetector(model_path=det_weights, device=device)
    lane_model = LaneSegmentationModel(model_path=lane_weights, device=device)
    drivable_model = DrivableAreaSegmentationModel(model_path=drivable_weights, device=device)

    policy_model = None
    if policy_weights and Path(policy_weights).exists():
        policy_model = __import__('policy.convlstm_model', fromlist=['ConvLSTMPolicy']).ConvLSTMPolicy(in_channels=3, hidden_dim=32, num_outputs=1)
        ckpt = torch.load(str(policy_weights), map_location=device)
        state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        policy_model.load_state_dict(state)
        policy_model.to(device)
        logger.info("Loaded policy checkpoint %s", policy_weights)
    else:
        logger.warning("No policy checkpoint found at %s; proceeding without policy", policy_weights)

    writer = None
    if output_video:
        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_window: deque = deque(maxlen=seq_len)
    # policy processing size
    policy_h, policy_w = CONFIG.POLICY_FRAME_SIZE

    frame_count = 0
    try:
        while True:
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.detect(frame)
            if debug_det:
                print(f"[DETECT] Frame {frame_count}: {len(dets)} detections")
            lane_mask = predict_lane(lane_model, frame)
            drivable_mask = predict_drivable(drivable_model, frame)

            out = draw_detections(frame, dets)
            out = overlay_masks(out, drivable_mask, color=(0, 255, 0), alpha=0.25)
            out = overlay_masks(out, lane_mask, color=(0, 0, 255), alpha=0.4)

            # prepare the policy tensor
            img_policy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_policy = cv2.resize(img_policy, (policy_w, policy_h))
            t = torch.from_numpy(img_policy.astype('float32') / 255.0).permute(2, 0, 1).to(device)
            frame_window.append(t)
            predicted_steer: Optional[float] = None
            if policy_model is not None and len(frame_window) >= seq_len:
                model_seq = torch.stack(list(frame_window), dim=0).unsqueeze(0)  # (1, T, C, H, W)
                model_seq = model_seq.to(device)
                policy_model.eval()
                with torch.no_grad():
                    steering_tensor = policy_model(model_seq)
                if debug_policy:
                    print(f"[POLICY_TENSOR] Frame {frame_count}: tensor shape={getattr(steering_tensor, 'shape', None)} values={steering_tensor.cpu().numpy() if hasattr(steering_tensor, 'cpu') else steering_tensor}")
                try:
                    steering_value = float(steering_tensor.squeeze().cpu().item())
                    # debug print per frame index
                    print(f"[POLICY] Frame {frame_count}: steering_value = {steering_value:.4f}")
                    predicted_steer = steering_value
                    out = draw_steering(out, predicted_steer)
                except Exception:
                    predicted_steer = None

            if writer is not None:
                writer.write(out)

            # show
            cv2.imshow("Realtime Overlay Demo", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", type=str, default=None, help="Input video file or camera index (default: camera 0)")
    parser.add_argument("--output-video", type=str, default="outputs/realtime_demo_out.mp4", help="Output annotated video path")
    parser.add_argument("--weights-det", type=str, default=f"{CONFIG.CHECKPOINT_DIR}/yolov8n.pt", help="Detector weights (TODO)")
    parser.add_argument("--weights-lane", type=str, default=f"{CONFIG.CHECKPOINT_DIR}/lane_unet.pth", help="Lane model weights (TODO)")
    parser.add_argument("--weights-drivable", type=str, default=f"{CONFIG.CHECKPOINT_DIR}/drivable_unet.pth", help="Drivable model weights (TODO)")
    parser.add_argument("--weights-policy", type=str, default=f"{POLICY_CHECKPOINT_DIR}/policy_best.pt", help="Policy weights (best checkpoint)")
    parser.add_argument("--seq-len", type=int, default=3, help="Sequence length for the driving policy")
    parser.add_argument("--device", type=str, default=None, help="Device to use: mps, cuda, or cpu (default autodetect)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max number of frames to process (quick test)")
    parser.add_argument("--debug-det", action="store_true", help="Print detection counts per frame")
    parser.add_argument("--debug-policy", action="store_true", help="Print raw policy tensor per frame")
    args = parser.parse_args()

    # call the run_demo function
    input_src = args.input_video if args.input_video else 0
    run_demo(
        input_video=input_src,
        output_video=args.output_video,
        seq_len=args.seq_len,
        device=args.device,
        weights_det=args.weights_det,
        weights_lane=args.weights_lane,
        weights_drivable=args.weights_drivable,
        weights_policy=args.weights_policy,
        debug_det=args.debug_det,
        debug_policy=args.debug_policy,
        max_frames=args.max_frames,
    )


VISUAL_STEERING_SCALE = 3.0  # display-only scale factor for the gauge


def draw_steering(frame: np.ndarray, steer: float, pos: tuple[int, int] = (50, 50)) -> np.ndarray:
    """Draw steering label and a simple left/right gauge.

    steer in [-1, 1] expected but the network may not be normalized; we scale
    gauge length by a small factor. Draw a line inside a circle pointing
    left or right to indicate the steering angle.
    """
    out = frame.copy()
    label = f"Steering: {steer:+.3f}"
    cv2.putText(out, label, (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # gauge
    center = (pos[0] + 60, pos[1])
    radius = 20
    cv2.circle(out, center, radius, (255, 255, 255), 2)
    # Keep numeric label as the true predicted float value, but scale the
    # gauge so small steering signals are visually amplified for readability.
    scaled = max(-1.0, min(1.0, steer * VISUAL_STEERING_SCALE))
    angle = float(scaled) * (math.pi / 4.0)  # scale to +/-45 degrees visually
    end = (int(center[0] + radius * math.cos(angle - math.pi / 2)), int(center[1] + radius * math.sin(angle - math.pi / 2)))
    cv2.line(out, center, end, (0, 255, 255), 2)
    return out


if __name__ == "__main__":
    main()
