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
  --batch_size    Number of frame pairs to process in one forward pass      (default: 1)
  --flow_fps      Target optical-flow frame rate (default: same as video)
  --video_fps     Source video frame rate                                   (default: 25)
  --use_trt       Convert model to TensorRT before processing
  --trt_engine    Path to TRT engine file (built/loaded automatically)
  --trt_fp16      Use FP16 precision when building the TRT engine
  --trt_workspace TRT builder workspace in GiB                             (default: 4)
  --no_skip       Recompute even if output files already exist
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
    _frame_to_tensor,
    _load_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_FLOW_SUBDIR = "optical_flow"
_FRAME_PATTERN = re.compile(r"^frame(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
_DEFAULT_BATCH_SIZE = 1
_DEFAULT_VIDEO_FPS = 25.0
_DEFAULT_TRT_WORKSPACE_GB = 4


def _detect_trt_input_size(frame_dir: str, splits: list, scale: int) -> tuple:
    """Read the first available frame and return the TRT input (H, W) after
    applying *scale* and padding both dims to the nearest multiple of 8.

    Returns ``(h_padded, w_padded)`` for use in :func:`build_sea_raft_engine`.
    """
    for split in splits:
        split_dir = os.path.join(frame_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for entry in sorted(os.listdir(split_dir)):
            clip_dir = os.path.join(split_dir, entry)
            if not os.path.isdir(clip_dir):
                continue
            for fname in sorted(os.listdir(clip_dir)):
                if _FRAME_PATTERN.match(fname):
                    img = cv2.imread(os.path.join(clip_dir, fname))
                    if img is not None:
                        h, w = img.shape[:2]
                        if scale != 0:
                            factor = 2.0 ** scale  # 0.5 for scale=-1
                            h = int(round(h * factor))
                            w = int(round(w * factor))
                        # Pad to multiple of 8
                        h_pad = ((h // 8) + (1 if h % 8 else 0)) * 8
                        w_pad = ((w // 8) + (1 if w % 8 else 0)) * 8
                        return h_pad, w_pad
    raise RuntimeError(
        "Cannot detect input size: no readable frames found in "
        f"'{frame_dir}' for splits {splits}"
    )


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


def _all_flow_files_exist(flow_dir: str, frame_names: list, flow_stride: int = 1) -> bool:
    """Return ``True`` iff every expected flow file is already present.

    With *flow_stride* > 1 only every *flow_stride*-th frame is stored,
    so only those frames are checked.
    """
    return all(
        os.path.isfile(os.path.join(flow_dir, os.path.splitext(n)[0] + ".npy"))
        for i, n in enumerate(frame_names)
        if i % flow_stride == 0
    )


@torch.no_grad()
def _compute_flow_batch(
    model,
    args_ns,
    t1_batch: torch.Tensor,
    t2_batch: torch.Tensor,
) -> np.ndarray:
    """Run SEA-RAFT on a batch of frame pairs and return all flow maps.

    Mirrors :func:`_compute_flow` from ``test_optical_flow`` but operates on
    an arbitrary-sized batch rather than a single pair.

    Args:
        model:     Loaded SEA-RAFT model (already on the target device).
        args_ns:   SEA-RAFT config namespace (``iters``, ``scale``, …).
        t1_batch:  Previous frames, shape ``[B, 3, H, W]``, float32.
        t2_batch:  Current  frames, shape ``[B, 3, H, W]``, float32.

    Returns:
        NumPy array of shape ``[B, H, W, 2]``, float32, in the original
        spatial resolution.
    """
    scale = getattr(args_ns, "scale", -1)

    if scale != 0:
        up = 2.0 ** scale
        t1s = F.interpolate(t1_batch, scale_factor=up, mode="bilinear", align_corners=False)
        t2s = F.interpolate(t2_batch, scale_factor=up, mode="bilinear", align_corners=False)
    else:
        t1s, t2s = t1_batch, t2_batch

    output = model(t1s, t2s, iters=args_ns.iters, test_mode=True)
    flow_final = output['flow'][-1]   # [B, 2, H_s, W_s]

    flow_final = output['flow'][-1]
    # info_final = output['info'][-1]

    if scale != 0:
        down = 0.5 ** scale
        # flow = (
        #     F.interpolate(flow_final, size=(t1.shape[2], t1.shape[3]), mode="bilinear", align_corners=False)
        #     * down
        # )
        flow_down = F.interpolate(flow_final, scale_factor=down, mode='bilinear', align_corners=False) * down
        # info_down = F.interpolate(info_final, scale_factor=down, mode='area')
    else:
        flow_down = flow_final

    # Return at SEA-RAFT native resolution (half-res when scale=-1).
    # Values are in scaled-pixel units; the loader rescales on upsample.
    # [B, 2, H_s, W_s] → [B, H_s, W_s, 2], stored as float16 to halve disk space.
    return flow_final.permute(0, 2, 3, 1).cpu().to(torch.float16).numpy()


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
    batch_size: int = _DEFAULT_BATCH_SIZE,
    flow_fps: float = None,
    video_fps: float = _DEFAULT_VIDEO_FPS,
    use_trt: bool = False,
    trt_engine_path: str = None,
    trt_fp16: bool = True,
    trt_workspace_gb: int = _DEFAULT_TRT_WORKSPACE_GB,
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
        batch_size:
            Number of consecutive frame pairs to stack into a single
            batched forward pass.  Values greater than 1 improve GPU
            utilisation at the cost of proportionally more VRAM.
            Defaults to ``1`` (original single-pair behaviour).
        use_trt:
            When ``True``, convert the loaded PyTorch model to a TensorRT
            engine before processing.  If *trt_engine_path* is provided and
            the file exists it is loaded directly (no build step).
        trt_engine_path:
            Path to save/load the serialised TRT ``.engine`` file.  When
            ``None`` a name is derived automatically from the sea_raft_dir,
            input resolution, and FP16 flag.
        trt_fp16:
            Build the TRT engine with FP16 precision (default ``True``).
        trt_workspace_gb:
            TRT builder workspace in GiB (default 4).
    """
    # ── device ────────────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    batch_size = max(1, batch_size)
    print(f"[INFO] Batch size:   {batch_size}")

    # ── FPS subsampling ───────────────────────────────────────────────────
    if flow_fps is not None and flow_fps > 0:
        flow_stride = max(1, round(video_fps / flow_fps))
        print(f"[INFO] Flow FPS:     {flow_fps:.1f}  (every {flow_stride} frame(s))")
    else:
        flow_stride = 1
        print(f"[INFO] Flow FPS:     all frames")

    # ── SEA-RAFT path & model ─────────────────────────────────────────────
    _add_sea_raft_to_path(sea_raft_dir)
    args_ns = _build_args_ns(config, iters, scale)
    model = _load_model(
        args_ns,
        model_url=model_url,
        model_path=model_path,
        device=device,
    )

    # ── optional TensorRT conversion ──────────────────────────────────────
    if use_trt:
        if device.type != "cuda":
            print("[WARN] TensorRT requires CUDA; --use_trt ignored on CPU.")
        else:
            from trt_export import SeaRaftTRTEngine, build_sea_raft_engine  # noqa: E402

            # Derive engine path automatically if not given
            if trt_engine_path is None:
                _scale_val = getattr(args_ns, "scale", -1)
                _prec = "fp16" if trt_fp16 else "fp32"
                trt_engine_path = os.path.join(
                    sea_raft_dir,
                    f"sea_raft_{_prec}_iters{args_ns.iters}.engine",
                )

            if not os.path.isfile(trt_engine_path):
                # Need to enumerate splits first to find a sample frame
                _splits_for_probe = splits
                if _splits_for_probe is None:
                    _splits_for_probe = sorted(
                        d for d in os.listdir(frame_dir)
                        if os.path.isdir(os.path.join(frame_dir, d))
                    )
                _scale_val = getattr(args_ns, "scale", -1)
                trt_h, trt_w = _detect_trt_input_size(
                    frame_dir, _splits_for_probe, _scale_val
                )
                print(
                    f"[INFO] TRT engine not found at {trt_engine_path}\n"
                    f"[INFO] Building engine: H={trt_h}, W={trt_w}, "
                    f"max_batch={batch_size}, fp16={trt_fp16}"
                )
                build_sea_raft_engine(
                    model, args_ns,
                    engine_path=trt_engine_path,
                    input_h=trt_h,
                    input_w=trt_w,
                    max_batch=batch_size,
                    fp16=trt_fp16,
                    workspace_gb=trt_workspace_gb,
                )
            else:
                print(f"[INFO] Loading existing TRT engine: {trt_engine_path}")

            model = SeaRaftTRTEngine(trt_engine_path, device=device)
            print("[INFO] Switched to TRT engine for inference.")

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

            if skip_existing and _all_flow_files_exist(flow_dir, frame_names, flow_stride):
                total_clips_skipped += 1
                continue

            os.makedirs(flow_dir, exist_ok=True)

            # ── streaming single-pass with t_prev carry-forward ────────────
            # Each frame is read exactly once.  t_prev is reused from the
            # previous iteration — no redundant disk reads.
            # Pending (t_prev, t_curr, out_path) tuples are flushed as a
            # batch when the buffer reaches batch_size or the clip ends.

            pending_t1: list = []   # previous-frame tensors  [1, 3, H, W]
            pending_t2: list = []   # current-frame  tensors  [1, 3, H, W]
            pending_out: list = []  # corresponding output paths

            def _flush() -> None:
                if not pending_t1:
                    return
                t1_b = torch.cat(pending_t1, dim=0)
                t2_b = torch.cat(pending_t2, dim=0)
                flows = _compute_flow_batch(model, args_ns, t1_b, t2_b)
                for flow, op in zip(flows, pending_out):
                    np.save(op, flow)
                pending_t1.clear()
                pending_t2.clear()
                pending_out.clear()

            t_prev = None
            _scale = getattr(args_ns, 'scale', -1)
            for frame_0idx, frame_name in enumerate(frame_names):
                frame_path = os.path.join(clip_dir, frame_name)
                stem = os.path.splitext(frame_name)[0]
                out_path = os.path.join(flow_dir, stem + ".npy")

                should_store = (frame_0idx % flow_stride == 0)

                frame_bgr = cv2.imread(frame_path)
                if frame_bgr is None:
                    print(f"[WARN] Could not read frame, skipping: {frame_path}")
                    _flush()        # commit any pending batch before the gap
                    t_prev = None
                    continue

                t_curr = _frame_to_tensor(frame_bgr, device)

                if t_prev is None:
                    # First usable frame — write zero flow at SEA-RAFT native
                    # resolution (half-res for scale=-1), stored as float16.
                    if should_store and not (skip_existing and os.path.isfile(out_path)):
                        h, w = frame_bgr.shape[:2]
                        if _scale != 0:
                            up = 2.0 ** _scale
                            hs, ws = int(round(h * up)), int(round(w * up))
                        else:
                            hs, ws = h, w
                        np.save(out_path, np.zeros((hs, ws, 2), dtype=np.float16))
                else:
                    if should_store and not (skip_existing and os.path.isfile(out_path)):
                        pending_t1.append(t_prev)
                        pending_t2.append(t_curr)
                        pending_out.append(out_path)
                        if len(pending_t1) >= batch_size:
                            _flush()

                t_prev = t_curr  # always advance — no re-read on next iteration

            _flush()  # commit any remaining pairs at end of clip

            # Free cached CUDA allocations accumulated during this clip so
            # the allocator's free-block list stays small across clips.
            if device.type == 'cuda':
                torch.cuda.empty_cache()

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
        "--batch_size", type=int, default=_DEFAULT_BATCH_SIZE,
        help="Number of frame pairs to process in one batched forward pass",
    )
    parser.add_argument(
        "--video_fps", type=float, default=_DEFAULT_VIDEO_FPS,
        help="Frame rate of the source videos (used to derive the flow stride)",
    )
    parser.add_argument(
        "--flow_fps", type=float, default=None,
        help="Target flow frame rate; e.g. 12.5 stores every other frame at 25 fps. "
             "None (default) stores flow for every frame.",
    )
    parser.add_argument(
        "--use_trt", action="store_true",
        help="Convert the model to a TensorRT engine before processing (CUDA only)",
    )
    parser.add_argument(
        "--trt_engine", default=None, metavar="PATH",
        help="Path to the TRT .engine file (auto-named inside --sea_raft_dir when omitted)",
    )
    parser.add_argument(
        "--trt_fp16", action="store_true", default=True,
        help="Build the TRT engine with FP16 precision (default: True)",
    )
    parser.add_argument(
        "--trt_no_fp16", dest="trt_fp16", action="store_false",
        help="Disable FP16 and build a FP32 TRT engine instead",
    )
    parser.add_argument(
        "--trt_workspace", type=int, default=_DEFAULT_TRT_WORKSPACE_GB,
        metavar="GB",
        help="TRT builder workspace in GiB",
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
        batch_size=args.batch_size,
        flow_fps=args.flow_fps,
        video_fps=args.video_fps,
        use_trt=args.use_trt,
        trt_engine_path=args.trt_engine,
        trt_fp16=args.trt_fp16,
        trt_workspace_gb=args.trt_workspace,
    )


if __name__ == "__main__":
    main()
