#!/usr/bin/env python3
"""
Precompute optical flow for all dataset clips and save per-frame flow maps.

For each consecutive frame pair (frame_{N-1}, frame_{N}) inside a clip the
SEA-RAFT optical flow is computed and written as a NumPy ``.npy`` file
(shape [H, W, 2], dtype float32) into a sub-folder that lives alongside the
original frames.  The output file is given the same stem as the source image:

    <frame_dir>/<split>/clip_<K>/frame<N>.jpg   – original frame
    <frame_dir>/<split>/clip_<K>/optical_flow/frame<N>.npy  – flow map

The first frame of every clip has no predecessor; a zero-flow map is stored
for it so that downstream code always finds exactly one flow file per frame.

Processing is resumable: clips whose every flow file is already present are
skipped unless ``--no_skip`` is passed.

Requirements
------------
Identical to ``test_optical_flow.py``:
    pip install -r requirements.txt
    git clone https://github.com/princeton-vl/SEA-RAFT.git SEA-RAFT

Basic usage
-----------
Process *train* and *val* splits, loading the model from HuggingFace::

    python process_optical_flow.py \\
        --frame_dir data/soccernetballanticipation/720p \\
        --splits train val

Point to a local checkpoint::

    python process_optical_flow.py \\
        --frame_dir data/soccernetballanticipation/720p \\
        --splits train val \\
        --model_path SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth \\
        --config    SEA-RAFT/config/eval/spring-M.json

Full argument reference
-----------------------
  --frame_dir     Root that contains split sub-folders (train/, val/, …)   (required)
  --splits        Which splits to process (default: all dirs in frame_dir)
  --sea_raft_dir  Path to the cloned SEA-RAFT repository                   (default: ./SEA-RAFT)
  --model_url     HuggingFace model ID                                      (default: MemorySlices/Tartan-C-T-TSKH-spring540x960-M)
  --model_path    Local .pth checkpoint; overrides --model_url
  --config        SEA-RAFT JSON config file
  --device        cpu | cuda                                                (default: auto-detect)
  --iters         RAFT refinement iterations                                (default: from config)
  --scale         Spatial scale exponent                                    (default: from config)
  --output_subdir Subdirectory name inside each clip for flow files         (default: optical_flow)
  --no_skip       Recompute even if output files already exist
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import shared helpers from test_optical_flow (same directory)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from test_optical_flow import (  # noqa: E402
    _DEFAULT_MODEL_URL,
    _add_sea_raft_to_path,
    _build_args_ns,
    _compute_flow,
    _frame_to_tensor,
    _load_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_FLOW_SUBDIR = "optical_flow"
_FRAME_PATTERN = re.compile(r"^frame(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sorted_frame_names(clip_dir: str) -> list:
    """Return frame filenames from *clip_dir* sorted by their numeric index.

    Only files that match ``frame<N>.<ext>`` (JPEG / PNG) are included.
    Files at other paths (e.g. ``720p.mp4``, label JSONs) are ignored.

    Args:
        clip_dir: Absolute path to a clip directory.

    Returns:
        List of filenames (not full paths) in ascending frame-number order.
        Empty list when the directory cannot be listed or contains no frames.
    """
    try:
        names = os.listdir(clip_dir)
    except OSError:
        return []
    pairs = [
        (int(m.group(1)), name)
        for name in names
        if (m := _FRAME_PATTERN.match(name))
    ]
    pairs.sort(key=lambda x: x[0])
    return [name for _, name in pairs]


def _all_flow_files_exist(flow_dir: str, frame_names: list) -> bool:
    """Return ``True`` iff every frame already has a corresponding flow file."""
    return all(
        os.path.isfile(os.path.join(flow_dir, os.path.splitext(n)[0] + ".npy"))
        for n in frame_names
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def precompute_optical_flow(
    frame_dir: str,
    sea_raft_dir: str = "SEA-RAFT",
    model_url: str = None,
    model_path: str = None,
    config: str = None,
    device: torch.device = None,
    iters: int = None,
    scale: int = None,
    splits: list = None,
    output_subdir: str = _DEFAULT_FLOW_SUBDIR,
    skip_existing: bool = True,
) -> None:
    """Precompute SEA-RAFT optical flow for all clips in a dataset split.

    Iterates every clip directory found under ``<frame_dir>/<split>/``,
    computes the forward optical flow between each consecutive frame pair, and
    writes the result as a ``float32`` NumPy array ``[H, W, 2]`` to::

        <frame_dir>/<split>/<clip>/optical_flow/<frame_stem>.npy

    The first frame of each clip (which has no predecessor) receives a
    zero-valued flow map of the same spatial resolution.

    Args:
        frame_dir:
            Root directory that contains the split sub-folders
            (e.g. ``data/soccernetballanticipation/720p``).  The expected
            layout is::

                <frame_dir>/<split>/clip_<N>/frame<M>.jpg

        sea_raft_dir:
            Path to the cloned SEA-RAFT repository.  The ``core/`` sub-
            directory is added to ``sys.path`` automatically.
        model_url:
            HuggingFace model identifier.  Ignored when *model_path* is set.
            Defaults to the spring-M Tartan checkpoint.
        model_path:
            Local ``.pth`` checkpoint.  When provided, *model_url* is ignored.
        config:
            Path to a SEA-RAFT JSON configuration file.  Values from the
            file override the built-in spring-M defaults.
        device:
            PyTorch device for inference.  Auto-detected (CUDA if available,
            otherwise CPU) when ``None``.
        iters:
            Number of RAFT refinement iterations.  Overrides the value in
            *config* or the built-in default when not ``None``.
        scale:
            Spatial scale exponent passed to SEA-RAFT (``-1`` → half
            resolution).  Overrides *config* / built-in default.
        splits:
            List of split names to process (e.g. ``["train", "val"]``).
            When ``None`` all immediate sub-directories of *frame_dir* that
            are directories are used.
        output_subdir:
            Name of the sub-directory created inside each clip folder to
            hold the ``.npy`` flow files.  Defaults to ``"optical_flow"``.
        skip_existing:
            When ``True`` (default), clips whose flow files are already
            fully present are skipped without re-running inference.
    """
    # ── device ────────────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── SEA-RAFT path & model ─────────────────────────────────────────────
    _add_sea_raft_to_path(sea_raft_dir)
    args_ns = _build_args_ns(config, iters, scale)
    model = _load_model(
        args_ns,
        model_url=model_url,
        model_path=model_path,
        device=device,
    )

    # ── enumerate splits ──────────────────────────────────────────────────
    if splits is None:
        try:
            splits = sorted(
                d for d in os.listdir(frame_dir)
                if os.path.isdir(os.path.join(frame_dir, d))
            )
        except OSError as exc:
            raise RuntimeError(
                f"Cannot list frame_dir '{frame_dir}': {exc}"
            ) from exc

    total_clips_processed = 0
    total_clips_skipped = 0

    for split in splits:
        split_dir = os.path.join(frame_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split directory not found, skipping: {split_dir}")
            continue

        # Collect clip directories, sorted numerically (clip_1 < clip_2 < …)
        clip_names = sorted(
            (d for d in os.listdir(split_dir)
             if os.path.isdir(os.path.join(split_dir, d))),
            key=lambda s: int(m.group()) if (m := re.search(r"\d+", s)) else 0,
        )
        print(f"\n[INFO] Split '{split}': {len(clip_names)} clip directory(ies) found.")

        for clip_name in tqdm(clip_names, desc=split, unit="clip"):
            clip_dir = os.path.join(split_dir, clip_name)
            flow_dir = os.path.join(clip_dir, output_subdir)

            frame_names = _sorted_frame_names(clip_dir)
            if not frame_names:
                # Clip directory has no extracted frames (e.g. only 720p.mp4)
                continue

            if skip_existing and _all_flow_files_exist(flow_dir, frame_names):
                total_clips_skipped += 1
                continue

            os.makedirs(flow_dir, exist_ok=True)

            # ── per-clip forward pass ──────────────────────────────────────
            t_prev = None
            for frame_name in frame_names:
                frame_path = os.path.join(clip_dir, frame_name)
                stem = os.path.splitext(frame_name)[0]
                out_path = os.path.join(flow_dir, stem + ".npy")

                # When resuming a partial clip, keep t_prev in sync without
                # re-running inference for frames that already have output.
                if skip_existing and os.path.isfile(out_path):
                    frame_bgr = cv2.imread(frame_path)
                    if frame_bgr is not None:
                        t_prev = _frame_to_tensor(frame_bgr, device)
                    continue

                frame_bgr = cv2.imread(frame_path)
                if frame_bgr is None:
                    print(f"[WARN] Could not read frame, skipping: {frame_path}")
                    t_prev = None
                    continue

                t_curr = _frame_to_tensor(frame_bgr, device)

                if t_prev is None:
                    # First frame — no predecessor; store zero flow
                    h, w = frame_bgr.shape[:2]
                    flow_np = np.zeros((h, w, 2), dtype=np.float32)
                else:
                    flow_np, _ = _compute_flow(model, args_ns, t_prev, t_curr)

                np.save(out_path, flow_np)
                t_prev = t_curr

            total_clips_processed += 1

    print(
        f"\n[INFO] Done. Processed {total_clips_processed} clip(s), "
        f"skipped {total_clips_skipped} already-complete clip(s)."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute SEA-RAFT optical flow for all dataset clips",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frame_dir", required=True,
        help="Root directory containing split sub-folders (e.g. data/soccernetballanticipation/720p)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Split names to process (default: all sub-directories of frame_dir)",
    )
    parser.add_argument(
        "--sea_raft_dir", default="SEA-RAFT",
        help="Path to the cloned SEA-RAFT repository",
    )
    parser.add_argument(
        "--model_url", default=_DEFAULT_MODEL_URL,
        help="HuggingFace model ID (used when --model_path is not given)",
    )
    parser.add_argument(
        "--model_path", default=None,
        help="Local .pth checkpoint; overrides --model_url",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a SEA-RAFT JSON config file",
    )
    parser.add_argument(
        "--device", default=None,
        help="Inference device: 'cpu' or 'cuda' (default: auto-detect)",
    )
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Number of RAFT refinement iterations (overrides config)",
    )
    parser.add_argument(
        "--scale", type=int, default=None,
        help="Spatial scale exponent for SEA-RAFT (overrides config)",
    )
    parser.add_argument(
        "--output_subdir", default=_DEFAULT_FLOW_SUBDIR,
        help="Sub-directory name inside each clip folder for .npy flow files",
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Recompute flow even if output files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device(args.device) if args.device else None

    precompute_optical_flow(
        frame_dir=args.frame_dir,
        sea_raft_dir=args.sea_raft_dir,
        model_url=args.model_url,
        model_path=args.model_path,
        config=args.config,
        device=device,
        iters=args.iters,
        scale=args.scale,
        splits=args.splits,
        output_subdir=args.output_subdir,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
