#!/usr/bin/env python3
"""
Visualise sample frames alongside their precomputed optical flow maps.

For each selected frame the script shows a side-by-side window:
    left  – original frame (JPEG)
    right – colour-coded optical flow (from .npy file in optical_flow/)

The very first frame of a clip has no predecessor so its flow map is all
zeros; those frames are skipped by default (--show_first to include them).

Usage
-----
Show 5 random frames from a clip::

    python visualize_optical_flow.py \\
        --clip_dir data/soccernetballanticipation/720p/train/clip_10

Show specific frames by number::

    python visualize_optical_flow.py \\
        --clip_dir data/soccernetballanticipation/720p/train/clip_10 \\
        --frames 5 10 50 100

Save output images instead of displaying them::

    python visualize_optical_flow.py \\
        --clip_dir data/soccernetballanticipation/720p/train/clip_10 \\
        --save_dir /tmp/flow_vis

Full argument reference
-----------------------
  --clip_dir      Path to the clip directory                     (required)
  --flow_subdir   Sub-directory name for .npy flow files         (default: optical_flow)
  --frames        Specific frame numbers to visualise
  --n_samples     Number of random frames to sample (when --frames not given, default: 5)
  --show_first    Include the first frame even though its flow is zero
  --save_dir      Save visualisations here instead of displaying interactively
  --sea_raft_dir  Path to SEA-RAFT repo (only needed for flow colourisation)
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reuse colour-wheel helper from test_optical_flow (same directory)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_DEFAULT_FLOW_SUBDIR = "optical_flow"
_FRAME_PATTERN = re.compile(r"^frame(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_frame_items(clip_dir: str) -> list:
    """Return ``[(frame_number, filename), ...]`` sorted by frame number."""
    items = []
    try:
        for name in os.listdir(clip_dir):
            m = _FRAME_PATTERN.match(name)
            if m:
                items.append((int(m.group(1)), name))
    except OSError:
        pass
    items.sort(key=lambda x: x[0])
    return items


def _flow_to_bgr_fallback(flow_np: np.ndarray) -> np.ndarray:
    """
    Colour-code *flow_np* ``[H, W, 2]`` using HSV mapping.

    Used when SEA-RAFT's ``flow_to_image`` is not available on sys.path.
    Hue encodes direction, value encodes magnitude (normalised per-image).
    """
    fx, fy = flow_np[..., 0], flow_np[..., 1]
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    angle = np.arctan2(fy, fx)  # radians in [-π, π]

    # Map angle → hue [0, 180]
    hue = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    # Normalise magnitude → saturation / value
    max_mag = magnitude.max()
    norm_mag = (magnitude / max_mag * 255).astype(np.uint8) if max_mag > 1e-6 else np.zeros_like(magnitude, dtype=np.uint8)

    hsv = np.stack([hue, norm_mag, norm_mag], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def _flow_to_bgr(flow_np: np.ndarray, sea_raft_dir: str | None) -> np.ndarray:
    """
    Colour-code *flow_np* ``[H, W, 2]`` → BGR uint8.

    Tries to use SEA-RAFT's colour wheel for best-quality visuals.
    Falls back to a simple HSV mapping when the library is unavailable.

    Args:
        flow_np:      Optical flow array of shape ``[H, W, 2]``, float32.
        sea_raft_dir: Path to the cloned SEA-RAFT repository, or ``None``.

    Returns:
        BGR colour image of shape ``[H, W, 3]``, uint8.
    """
    if sea_raft_dir is not None:
        core_dir = os.path.join(sea_raft_dir, "core")
        if os.path.isdir(core_dir) and core_dir not in sys.path:
            sys.path.insert(0, core_dir)
    try:
        from utils.flow_viz import flow_to_image  # noqa: PLC0415
        return flow_to_image(flow_np, convert_to_bgr=True)
    except ImportError:
        return _flow_to_bgr_fallback(flow_np)


def _make_side_by_side(frame_bgr: np.ndarray, flow_bgr: np.ndarray, frame_num: int) -> np.ndarray:
    """
    Combine *frame_bgr* and *flow_bgr* into a single labelled image.

    Resizes *flow_bgr* to match the original frame's spatial dimensions if
    they differ (should not normally happen), then stacks horizontally and
    adds text labels near the top.

    Args:
        frame_bgr: Original frame, shape ``[H, W, 3]``, BGR uint8.
        flow_bgr:  Colourised flow, shape ``[H, W, 3]``, BGR uint8.
        frame_num: Source frame number (shown in the window / file title).

    Returns:
        Side-by-side image of shape ``[H, W*2, 3]``, BGR uint8.
    """
    h, w = frame_bgr.shape[:2]
    if flow_bgr.shape[:2] != (h, w):
        flow_bgr = cv2.resize(flow_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    canvas = np.hstack([frame_bgr, flow_bgr])

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale, thick, colour = 0.65, 1, (220, 220, 220)
    shadow = (20, 20, 20)
    for text, cx in [
        (f"Original  frame {frame_num}", w // 2),
        (f"Optical Flow  frame {frame_num}", w + w // 2),
    ]:
        (tw, th), _ = cv2.getTextSize(text, font, text_scale, thick)
        x, y = cx - tw // 2, th + 8
        cv2.putText(canvas, text, (x + 1, y + 1), font, text_scale, shadow, thick, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), font, text_scale, colour, thick, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Core visualisation function
# ---------------------------------------------------------------------------

def visualise_samples(
    clip_dir: str,
    flow_subdir: str = _DEFAULT_FLOW_SUBDIR,
    frames: list = None,
    n_samples: int = 5,
    show_first: bool = False,
    save_dir: str = None,
    sea_raft_dir: str = None,
) -> None:
    """Visualise original frames paired with their precomputed flow maps.

    Loads ``.npy`` flow files that were written by ``process_optical_flow.py``
    and displays (or saves) each original frame next to its colourised flow.

    Args:
        clip_dir:
            Path to a single clip directory that contains both the JPEG
            frames and the ``optical_flow/`` sub-folder produced by
            ``process_optical_flow.py``.
        flow_subdir:
            Name of the sub-directory inside *clip_dir* that holds the
            ``.npy`` flow files.  Defaults to ``"optical_flow"``.
        frames:
            Explicit list of integer frame numbers to visualise.  When
            ``None``, *n_samples* frames are chosen uniformly from those
            that have precomputed flow.
        n_samples:
            Number of frames to sample when *frames* is ``None``.
        show_first:
            When ``True``, include the first frame even though its flow
            map is all-zero (no predecessor exists).  Defaults to
            ``False``.
        save_dir:
            Directory to write PNG visualisations to instead of showing
            interactive windows.  Created if it does not exist.  When
            ``None`` (default), ``cv2.imshow`` is used.
        sea_raft_dir:
            Path to the cloned SEA-RAFT repository.  Provides a higher-
            quality colour wheel for flow visualisation.  Falls back to
            an HSV mapping when ``None`` or the library cannot be loaded.
    """
    flow_dir = os.path.join(clip_dir, flow_subdir)
    if not os.path.isdir(flow_dir):
        raise FileNotFoundError(
            f"Flow directory not found: '{flow_dir}'\n"
            "Run process_optical_flow.py first to precompute the flow maps."
        )

    # ── collect available frames that have a flow file ─────────────────────
    all_items = _sorted_frame_items(clip_dir)  # [(frame_num, filename), ...]
    if not all_items:
        raise RuntimeError(f"No frame images found in '{clip_dir}'.")

    first_num = all_items[0][0]
    available = []
    for frame_num, frame_name in all_items:
        stem = os.path.splitext(frame_name)[0]
        flow_path = os.path.join(flow_dir, stem + ".npy")
        if not os.path.isfile(flow_path):
            continue
        is_first = frame_num == first_num
        if is_first and not show_first:
            continue
        available.append((frame_num, frame_name, flow_path))

    if not available:
        msg = (
            f"No frames with precomputed flow found in '{clip_dir}'."
            if show_first
            else (
                f"No non-first frames with precomputed flow found in '{clip_dir}'. "
                "Use --show_first to include the first frame."
            )
        )
        raise RuntimeError(msg)

    # ── select frames ──────────────────────────────────────────────────────
    if frames is not None:
        target_set = set(frames)
        selected = [item for item in available if item[0] in target_set]
        missing = target_set - {item[0] for item in selected}
        if missing:
            print(f"[WARN] Requested frame(s) not available: {sorted(missing)}")
    else:
        step = max(1, len(available) // n_samples)
        selected = available[::step][:n_samples]

    if not selected:
        raise RuntimeError("No frames to display after filtering.")

    print(f"[INFO] Visualising {len(selected)} frame(s) from '{clip_dir}'")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ── per-frame render loop ──────────────────────────────────────────────
    for frame_num, frame_name, flow_path in selected:
        frame_path = os.path.join(clip_dir, frame_name)
        frame_bgr = cv2.imread(frame_path)
        if frame_bgr is None:
            print(f"[WARN] Cannot read image: {frame_path}, skipping.")
            continue

        flow_np = np.load(flow_path)  # [H, W, 2] float32
        flow_bgr = _flow_to_bgr(flow_np, sea_raft_dir)

        canvas = _make_side_by_side(frame_bgr, flow_bgr, frame_num)

        if save_dir:
            out_name = f"flow_vis_frame{frame_num}.png"
            out_path = os.path.join(save_dir, out_name)
            cv2.imwrite(out_path, canvas)
            print(f"[INFO] Saved → {out_path}")
        else:
            title = f"Frame {frame_num} | press any key for next, q to quit"
            cv2.imshow(title, canvas)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                print("[INFO] Quit by user.")
                break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise precomputed optical flow alongside original frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clip_dir", required=True,
        help="Path to the clip directory (must contain extracted frames and optical_flow/)",
    )
    parser.add_argument(
        "--flow_subdir", default=_DEFAULT_FLOW_SUBDIR,
        help="Sub-directory name inside clip_dir that holds the .npy flow files",
    )
    parser.add_argument(
        "--frames", type=int, nargs="+", default=None,
        help="Specific frame numbers to visualise (e.g. --frames 5 10 50)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=5,
        help="Number of frames to show when --frames is not specified",
    )
    parser.add_argument(
        "--show_first", action="store_true",
        help="Include the first frame (its flow is zero — no predecessor)",
    )
    parser.add_argument(
        "--save_dir", default=None,
        help="Save visualisations as PNG files to this directory instead of showing windows",
    )
    parser.add_argument(
        "--sea_raft_dir", default="SEA-RAFT",
        help="Path to the cloned SEA-RAFT repo (for higher-quality flow colour wheel)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    visualise_samples(
        clip_dir=args.clip_dir,
        flow_subdir=args.flow_subdir,
        frames=args.frames,
        n_samples=args.n_samples,
        show_first=args.show_first,
        save_dir=args.save_dir,
        sea_raft_dir=args.sea_raft_dir,
    )


if __name__ == "__main__":
    main()
