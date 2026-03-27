#!/usr/bin/env python3
"""
Optical flow test script using SEA-RAFT.

Processes a video file frame-by-frame with the SEA-RAFT optical flow model
and writes a side-by-side output video:
    left half  – original frame
    right half – colour-coded optical flow visualisation

Requirements
------------
Install Python dependencies::

    pip install -r requirements.txt

Clone SEA-RAFT (only needed once)::

    git clone https://github.com/princeton-vl/SEA-RAFT.git SEA-RAFT

Basic usage
-----------
Load the model from HuggingFace (no local checkpoint needed)::

    python test_optical_flow.py \\
        --video  input.mp4 \\
        --output output_flow.mp4

Point to a local model checkpoint and config::

    python test_optical_flow.py \\
        --video  input.mp4 \\
        --output output_flow.mp4 \\
        --model_path SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth \\
        --config  SEA-RAFT/config/eval/spring-M.json

Full list of arguments
----------------------
  --video         Path to the input video file                 (required)
  --output        Path for the output side-by-side video       (default: <input>_flow.mp4)
  --sea_raft_dir  Path to the cloned SEA-RAFT repository       (default: ./SEA-RAFT)
  --model_url     HuggingFace model ID                         (default: MemorySlices/Tartan-C-T-TSKH-spring540x960-M)
  --model_path    Local model checkpoint (.pth); overrides URL
  --config        Path to a SEA-RAFT JSON config file
  --device        Inference device: cpu | cuda                 (default: auto-detect)
  --iters         Number of RAFT refinement iterations         (default: from config / 4)
  --scale         Spatial scale exponent passed to SEA-RAFT    (default: from config / -1)
  --frame_skip    Process every N-th frame pair                (default: 1 = every frame)
  --max_frames    Stop after this many output frames (0 = all) (default: 0)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Default SEA-RAFT model configuration (spring-M)
# Used when no --config file is provided.
# ---------------------------------------------------------------------------
_DEFAULT_CFG = {
    "use_var": True,
    "var_min": 0,
    "var_max": 10,
    "pretrain": "resnet34",
    "initial_dim": 64,
    "block_dims": [64, 128, 256],
    "radius": 4,
    "dim": 128,
    "num_blocks": 2,
    "iters": 4,
    "scale": -1,
    "epsilon": 1e-8,
    "dropout": 0,
}

_DEFAULT_MODEL_URL = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_sea_raft_to_path(sea_raft_dir: str) -> None:
    """Insert the SEA-RAFT *core* directory at the front of sys.path."""
    core_dir = os.path.join(sea_raft_dir, "core")
    if not os.path.isdir(core_dir):
        raise FileNotFoundError(
            f"SEA-RAFT core directory not found at '{core_dir}'.\n"
            "Please clone the repository first:\n"
            f"    git clone https://github.com/princeton-vl/SEA-RAFT.git {sea_raft_dir}"
        )
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)


def _build_args_ns(cfg_path: str | None, cli_iters: int | None, cli_scale: int | None):
    """
    Return an argparse.Namespace with SEA-RAFT model arguments.

    Priority: CLI overrides > JSON config > built-in defaults.

    Args:
        cfg_path:  Optional path to a SEA-RAFT JSON config file.  When
                   provided its values are merged on top of the built-in
                   spring-M defaults.
        cli_iters: If not None, overrides the ``iters`` field from the
                   config (number of RAFT refinement iterations).
        cli_scale: If not None, overrides the ``scale`` field from the
                   config (spatial scale exponent; -1 means half resolution).

    Returns:
        argparse.Namespace populated with all required SEA-RAFT model fields.
    """
    cfg = dict(_DEFAULT_CFG)
    if cfg_path is not None:
        with open(cfg_path) as fh:
            cfg.update(json.load(fh))
    if cli_iters is not None:
        cfg["iters"] = cli_iters
    if cli_scale is not None:
        cfg["scale"] = cli_scale
    return argparse.Namespace(**cfg)


def _load_model(args_ns: argparse.Namespace, model_url: str | None, model_path: str | None, device: torch.device):
    """Load a SEA-RAFT RAFT model either from HuggingFace or a local checkpoint."""
    from raft import RAFT  # noqa: PLC0415  (SEA-RAFT must be on sys.path)
    from utils.utils import load_ckpt  # noqa: PLC0415

    if model_path is not None:
        print(f"[INFO] Loading model from local checkpoint: {model_path}")
        model = RAFT(args_ns)
        load_ckpt(model, model_path)
    else:
        url = model_url or _DEFAULT_MODEL_URL
        print(f"[INFO] Loading model from HuggingFace: {url}")
        model = RAFT.from_pretrained(url, args=args_ns)

    model = model.to(device)
    model.eval()
    return model


def _frame_to_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a BGR uint8 HxWx3 NumPy frame to a float32 1x3xHxW tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


@torch.no_grad()
def _compute_flow(model, args_ns, t1: torch.Tensor, t2: torch.Tensor) -> np.ndarray:
    """
    Run SEA-RAFT on a pair of frames and return the optical flow as a
    NumPy array of shape [H, W, 2] in the original frame resolution.
    """
    scale = getattr(args_ns, "scale", -1)

    if scale != 0:
        up = 2.0 ** scale
        t1s = F.interpolate(t1, scale_factor=up, mode="bilinear", align_corners=False)
        t2s = F.interpolate(t2, scale_factor=up, mode="bilinear", align_corners=False)
    else:
        t1s, t2s = t1, t2

    output = model(t1s, t2s, iters=args_ns.iters, test_mode=True)
    flow_final = output['flow'][-1]
    info_final = output['info'][-1]

    if scale != 0:
        down = 0.5 ** scale
        # flow = (
        #     F.interpolate(flow_final, size=(t1.shape[2], t1.shape[3]), mode="bilinear", align_corners=False)
        #     * down
        # )
        flow_down = F.interpolate(flow_final, scale_factor=down, mode='bilinear', align_corners=False) * down
        info_down = F.interpolate(info_final, scale_factor=down, mode='area')
    else:
        flow_down, info_down = flow_final, info_final
    
    return (flow_down[0].permute(1, 2, 0).cpu().numpy(),  # [H, W, 2]
            info_down[0].permute(1, 2, 0).cpu().numpy())  # [H, W, C]


def _flow_to_bgr(flow_np: np.ndarray) -> np.ndarray:
    """Convert [H, W, 2] optical flow to a BGR colour image using SEA-RAFT's colour wheel."""
    from utils.flow_viz import flow_to_image  # noqa: PLC0415

    return flow_to_image(flow_np, convert_to_bgr=True)


def _add_side_labels(frame: np.ndarray, width: int) -> np.ndarray:
    """
    Overlay text labels on a side-by-side frame in-place.

    Adds 'Original' centred over the left half and 'Optical Flow' centred
    over the right half, near the top of the image.

    Args:
        frame: Combined side-by-side image of shape [H, W*2, 3] (BGR uint8).
        width: Width of a single (non-doubled) frame in pixels.

    Returns:
        The input frame with text labels drawn on it (same array, modified
        in-place).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, colour = 0.65, 1, (220, 220, 220)
    shadow = (20, 20, 20)
    for text, cx in [("Original", width // 2), ("Optical Flow", width + width // 2)]:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x, y = cx - tw // 2, th + 8
        cv2.putText(frame, text, (x + 1, y + 1), font, scale, shadow, thick, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_video(
    input_path: str,
    output_path: str,
    model,
    args_ns,
    device: torch.device,
    frame_skip: int = 1,
    max_frames: int = 0,
) -> None:
    """
    Read *input_path* and write *output_path*.

    Each output frame is [original | optical-flow] placed side by side.
    The flow is computed between the previous frame and the current frame.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Input:  {input_path}")
    print(f"[INFO] Size:   {width}×{height}  FPS: {fps:.2f}  Frames: {total}")
    print(f"[INFO] Output: {output_path}")

    out_w, out_h = width * 2, height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Cannot create output video writer for: {output_path}")

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        writer.release()
        raise IOError("Cannot read the first frame of the video.")

    # Tensor of the previous frame (kept on device between iterations)
    t_prev = _frame_to_tensor(prev_frame, device)

    written = 0
    src_idx = 0  # index of frames read from source (excluding frame 0)
    pbar = tqdm(total=(total - 1) if total > 1 else None, desc="Frames", unit="frame")

    try:
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            src_idx += 1

            if src_idx % frame_skip != 0:
                # Still advance t_prev so flow stays temporally consistent
                t_prev = _frame_to_tensor(curr_frame, device)
                pbar.update(1)
                continue

            t_curr = _frame_to_tensor(curr_frame, device)

            # --- Optical flow ---
            flow_np, info_np = _compute_flow(model, args_ns, t_prev, t_curr)
            flow_bgr = _flow_to_bgr(flow_np)

            # Resize flow visualisation to match original frame size (safety)
            if flow_bgr.shape[0] != height or flow_bgr.shape[1] != width:
                flow_bgr = cv2.resize(flow_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

            # --- Side-by-side composition ---
            side_by_side = np.hstack([prev_frame, flow_bgr])
            _add_side_labels(side_by_side, width)
            writer.write(side_by_side)

            t_prev = t_curr
            prev_frame = curr_frame
            written += 1
            pbar.update(1)

            if max_frames > 0 and written >= max_frames:
                break
    finally:
        pbar.close()
        cap.release()
        writer.release()

    print(f"[INFO] Wrote {written} frames → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SEA-RAFT optical flow – side-by-side video visualisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument(
        "--output", default=None,
        help="Output video file path (default: <input_stem>_flow.mp4)"
    )
    parser.add_argument(
        "--sea_raft_dir", default="SEA-RAFT",
        help="Path to the cloned SEA-RAFT repository"
    )
    parser.add_argument(
        "--model_url", default=_DEFAULT_MODEL_URL,
        help="HuggingFace model ID used when --model_path is not given"
    )
    parser.add_argument(
        "--model_path", default=None,
        help="Local .pth checkpoint; if set, overrides --model_url"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a SEA-RAFT JSON config file"
    )
    parser.add_argument(
        "--device", default=None,
        help="Inference device: 'cpu' or 'cuda' (default: auto-detect)"
    )
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Number of RAFT refinement iterations (overrides config)"
    )
    parser.add_argument(
        "--scale", type=int, default=None,
        help="Spatial scale exponent for SEA-RAFT (overrides config)"
    )
    parser.add_argument(
        "--frame_skip", type=int, default=1,
        help="Process every N-th consecutive frame pair (1 = every frame)"
    )
    parser.add_argument(
        "--max_frames", type=int, default=0,
        help="Maximum number of output frames to write (0 = unlimited)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Output path ---
    if args.output is None:
        stem, _ = os.path.splitext(args.video)
        args.output = stem + "_flow.mp4"

    # --- Device ---
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # --- SEA-RAFT path setup ---
    _add_sea_raft_to_path(args.sea_raft_dir)

    # --- Build model args namespace ---
    model_args = _build_args_ns(args.config, args.iters, args.scale)

    # --- Load model ---
    model = _load_model(
        model_args,
        model_url=args.model_url,
        model_path=args.model_path,
        device=device,
    )

    # --- Process video ---
    process_video(
        input_path=args.video,
        output_path=args.output,
        model=model,
        args_ns=model_args,
        device=device,
        frame_skip=max(1, args.frame_skip),
        max_frames=max(0, args.max_frames),
    )


if __name__ == "__main__":
    main()
