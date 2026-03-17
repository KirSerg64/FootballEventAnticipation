"""sam3_wrapper.py — SAM3 video-inference wrapper for football analysis.

This module provides :class:`Sam3SegmentationTracker`, a drop-in replacement
for :class:`~segmentation_tracking.segmentation_model.SegmentationTracker`
that uses **SAM 3** (Segment Anything with Concepts) instead of the legacy
SAM 2 + YOLO-seeded pipeline.

Key advantages of SAM 3 over SAM 2 in this context
----------------------------------------------------
* **Text-prompt driven** — objects are identified by natural-language
  descriptions (e.g. ``"football player"``, ``"sports ball"``,
  ``"football pitch"``).  No YOLO-based seeding is required.
* **Open-vocabulary** — easily adapts to other sports without retraining.
* **Field segmentation** — a dedicated text prompt can segment the pitch
  boundary, making it available to downstream consumers (e.g. homography
  estimation, bird's-eye view transforms).

Memory-efficient frame-by-frame processing
------------------------------------------
The original design passed the entire video file to SAM3's ``init_state``
/ ``start_session`` call, which caused SAM3 to pre-load **all** frames
into GPU VRAM simultaneously — leading to ``CUDA out of memory`` errors on
GPUs with limited VRAM.

The rewritten implementation avoids this by:

1. Decoding the video frame-by-frame with OpenCV (CPU side, no GPU).
2. Writing each frame as a JPEG image into a temporary directory.
3. Passing the *frame directory* (not the video file) to SAM3.  SAM3/SAM2
   supports both video files and image directories.
4. Setting ``offload_video_to_cpu=True`` in ``init_state`` (where
   supported), so SAM3 reads exactly one frame from disk into GPU VRAM at
   a time instead of holding all frames on GPU.

GPU memory footprint is therefore::

    model weights  +  tracking state  +  ~1 frame
    (~3–6 GiB)        (small)            (~10 MB for 1080p)

instead of::

    model weights  +  tracking state  +  ALL frames
    (~3–6 GiB)        (small)            (>> 1 GiB for long videos)

Model weights
-------------
Download the SAM 3 checkpoint from the official Hugging Face repository
(gated — requires approval):

    https://huggingface.co/facebook/sam3

Place the ``.pt`` file at::

    weights/sam3/sam3.pt

(This path is the default ``sam3_model_path`` for
:class:`Sam3SegmentationTracker`.)

Package dependency
------------------
SAM 3 must be installed separately:

    git clone https://github.com/facebookresearch/sam3
    cd sam3
    pip install -e .

If the package is not installed the class still loads but
:meth:`Sam3SegmentationTracker.process_video` will raise a clear
:class:`ImportError` at call time.

Usage example
-------------
::

    from segmentation_tracking import Sam3SegmentationTracker

    tracker = Sam3SegmentationTracker(
        sam3_model_path="weights/sam3/sam3.pt",
        player_text_prompt="football player",
        ball_text_prompt="sports ball",
        field_text_prompt="football pitch",
    )
    results = tracker.process_video("match.mp4")
    for r in results:
        print(r.frame_index, len(r.player_ids), r.ball_center, r.field_mask is not None)
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import cv2
import numpy as np

from segmentation_tracking.segmentation_model import SegmentationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default text prompts — can be overridden at construction time
# ---------------------------------------------------------------------------
_DEFAULT_PLAYER_PROMPT: str = "football player"
_DEFAULT_BALL_PROMPT: str = "sports ball"
_DEFAULT_FIELD_PROMPT: str | None = None   # field segmentation disabled by default
_DEFAULT_MODEL_PATH: str = "weights/sam3/sam3.pt"

# Minimum SAM3 object score to accept a detection.
_DEFAULT_SCORE_THRESH: float = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_tensor_to_numpy(mask_t: Any) -> np.ndarray:
    """Convert a SAM3 mask tensor to a uint8 numpy mask (0 / 255)."""
    import torch  # local import to keep module importable without torch

    if isinstance(mask_t, torch.Tensor):
        arr = mask_t.squeeze().cpu().float().numpy()
    else:
        arr = np.asarray(mask_t, dtype=np.float32)
    # SAM3 masks are logits (>0 = foreground) or probability-like [0, 1]
    binary = (arr > 0).astype(np.uint8) * 255
    return binary


def _mask_to_bbox(mask: np.ndarray) -> np.ndarray | None:
    """Return the tight bounding box ``[x1, y1, x2, y2]`` for *mask*."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _union_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Merge a list of binary masks into a single union mask."""
    out = np.zeros(shape, dtype=np.uint8)
    for m in masks:
        mh, mw = m.shape[:2]
        if (mh, mw) != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.maximum(out, m)
    return out


# ---------------------------------------------------------------------------
# Sam3SegmentationTracker
# ---------------------------------------------------------------------------

class Sam3SegmentationTracker:
    """SAM3-based player segmentation, ball detection, and field segmentation.

    This class mirrors the public API of
    :class:`~segmentation_tracking.segmentation_model.SegmentationTracker`
    (specifically :meth:`process_video`) so it can be used as a drop-in
    replacement in the main pipeline.

    The tracker runs **separate SAM3 inference sessions** for each semantic
    category (players, ball, field), then merges results into the canonical
    :class:`~segmentation_tracking.segmentation_model.SegmentationResult`
    data structure.

    Memory-efficient design
    -----------------------
    To avoid GPU out-of-memory errors caused by pre-loading the entire video
    into VRAM, :meth:`process_video` first extracts all video frames as JPEG
    images into a temporary directory using OpenCV (CPU-only).  SAM3 is then
    pointed at the frame directory and configured to read frames from disk
    one at a time (``offload_video_to_cpu=True``), so GPU VRAM usage stays
    proportional to the *model size* rather than the *video length*.

    Parameters
    ----------
    sam3_model_path:
        Path to the SAM3 ``.pt`` checkpoint file.  Default:
        ``"weights/sam3/sam3.pt"``.
    device:
        Torch device string: ``"cuda"`` or ``"cpu"``.
    player_text_prompt:
        Text description of the objects to track as players.  Default:
        ``"football player"``.
    ball_text_prompt:
        Text description of the ball.  Pass *None* to disable SAM3-based
        ball detection (in that case ``ball_mask`` / ``ball_center`` will
        always be *None*).  Default: ``"sports ball"``.
    field_text_prompt:
        Text description of the playing field.  Pass *None* (default) to
        disable field segmentation.  When set (e.g. ``"football pitch"``),
        :attr:`~segmentation_tracking.segmentation_model.SegmentationResult.field_mask`
        is populated for every frame.
    score_threshold:
        Minimum SAM3 object confidence score to accept a detection.
        Default ``0.30``.
    use_float16:
        When *True*, the SAM3 model weights are converted to float16
        immediately after loading, roughly halving the GPU VRAM footprint
        (~6–8 GB → ~3–4 GB for a standard SAM3 checkpoint).  This is
        especially useful when the GPU is shared with other models (pose
        estimator, ball tracker, etc.).  There is a tiny risk of numerical
        precision loss in rare edge cases, but in practice the segmentation
        quality is indistinguishable from float32.  Default ``False``.
    frame_jpeg_quality:
        JPEG quality (1–100) used when writing temporary frame images to
        disk during :meth:`process_video`.  Higher values preserve more
        detail but use more disk space.  Default ``90``.
    """

    def __init__(
        self,
        sam3_model_path: str = _DEFAULT_MODEL_PATH,
        device: str = "cuda",
        player_text_prompt: str = _DEFAULT_PLAYER_PROMPT,
        ball_text_prompt: str | None = _DEFAULT_BALL_PROMPT,
        field_text_prompt: str | None = _DEFAULT_FIELD_PROMPT,
        score_threshold: float = _DEFAULT_SCORE_THRESH,
        use_float16: bool = False,
        frame_jpeg_quality: int = 90,
    ) -> None:
        self.sam3_model_path = sam3_model_path
        self.device = device
        self.player_text_prompt = player_text_prompt
        self.ball_text_prompt = ball_text_prompt
        self.field_text_prompt = field_text_prompt
        self.score_threshold = score_threshold
        self.use_float16 = use_float16
        self.frame_jpeg_quality = int(max(1, min(100, frame_jpeg_quality)))

        # Lazy-loaded predictor (shared across calls)
        self._predictor: Any = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_predictor(self) -> Any:
        """Lazily load and return the SAM3 video predictor."""
        if self._predictor is not None:
            return self._predictor

        try:
            from sam3.model.sam3_video_predictor import Sam3VideoPredictor
        except ImportError as exc:
            raise ImportError(
                "The 'sam3' package is not installed.  Install it with:\n\n"
                "    git clone https://github.com/facebookresearch/sam3\n"
                "    cd sam3\n"
                "    pip install -e .\n\n"
                "Then download the model weights and place them at:\n"
                f"    {self.sam3_model_path}\n"
            ) from exc

        if not os.path.isfile(self.sam3_model_path):
            logger.warning(
                "SAM3 checkpoint not found at '%s'.  "
                "The predictor will attempt to load without explicit weights "
                "(may fail or use default pretrained weights).",
                self.sam3_model_path,
            )
            checkpoint_path = None
        else:
            checkpoint_path = self.sam3_model_path

        # ------------------------------------------------------------------
        # Pre-load memory management
        # ------------------------------------------------------------------
        # Free any cached (but not actively used) GPU memory so SAM3 can
        # get a contiguous allocation.  This is especially important when
        # other models (pose estimator, ball tracker, …) have already been
        # loaded and left fragmented allocator state.
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _free_before = torch.cuda.mem_get_info()[0] / 1024 ** 3
            logger.info(
                "GPU memory before SAM3 load: %.2f GiB free / %.2f GiB total",
                _free_before,
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            )

        # Sam3VideoPredictor has no 'device' parameter: it always calls .cuda()
        # internally (requires a CUDA-capable GPU).  If a non-CUDA device was
        # requested we log a warning but proceed — the SAM3 model itself decides
        # where to place its weights.
        if self.device != "cuda":
            logger.warning(
                "Sam3VideoPredictor always loads on CUDA internally. "
                "The requested device '%s' cannot be honoured.",
                self.device,
            )

        logger.info(
            "Loading SAM3 video predictor (checkpoint_path=%s, float16=%s)",
            checkpoint_path,
            self.use_float16,
        )

        try:
            self._predictor = Sam3VideoPredictor(checkpoint_path=checkpoint_path)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as oom_exc:
            # Provide a more actionable error message.
            msg = str(oom_exc)
            if "out of memory" in msg.lower() or "OutOfMemory" in type(oom_exc).__name__:
                if torch.cuda.is_available():
                    free_gib = torch.cuda.mem_get_info()[0] / 1024 ** 3
                    total_gib = (
                        torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                    )
                else:
                    free_gib = total_gib = 0.0
                raise RuntimeError(
                    f"CUDA out of memory while loading SAM3 "
                    f"(GPU has {free_gib:.1f} GiB free out of {total_gib:.1f} GiB total).\n\n"
                    "Suggested mitigations (in order of ease):\n"
                    "  1. Add --sam3_float16 to halve the model's VRAM footprint.\n"
                    "  2. Ensure no other GPU-heavy processes are running.\n"
                    "  3. Use a GPU with more VRAM (SAM3 needs ~6 GiB in float32, "
                    "~3 GiB in float16).\n"
                ) from oom_exc
            raise

        # ------------------------------------------------------------------
        # Optional float16 conversion
        # ------------------------------------------------------------------
        # Converting model weights to float16 roughly halves GPU VRAM usage.
        # We do this *after* a successful load because SAM3 always initialises
        # in float32 internally (.cuda()).
        if self.use_float16 and hasattr(self._predictor, "model"):
            try:
                self._predictor.model.half()
                logger.info(
                    "SAM3 model converted to float16 — "
                    "GPU VRAM usage is approximately halved."
                )
            except Exception as half_exc:
                logger.warning(
                    "Could not convert SAM3 model to float16 (%s); "
                    "continuing in float32.",
                    half_exc,
                )
        elif self.use_float16:
            logger.warning(
                "use_float16=True but the SAM3 predictor has no 'model' attribute; "
                "float16 conversion skipped."
            )

        if torch.cuda.is_available():
            _free_after = torch.cuda.mem_get_info()[0] / 1024 ** 3
            logger.info(
                "GPU memory after SAM3 load: %.2f GiB free / %.2f GiB total",
                _free_after,
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
            )

        return self._predictor

    # ------------------------------------------------------------------
    # Single-category SAM3 inference pass
    # ------------------------------------------------------------------

    def _run_sam3_for_prompt(
        self,
        frames_dir: str,
        text_prompt: str,
        num_frames: int,
    ) -> dict[int, list[tuple[int, np.ndarray]]]:
        """Run SAM3 for one text prompt over the pre-extracted frame directory.

        Parameters
        ----------
        frames_dir:
            Path to the temporary directory containing JPEG frame images
            named ``000000.jpg``, ``000001.jpg``, … as written by
            :meth:`_extract_frames_to_dir`.
        text_prompt:
            Natural-language description of the objects to segment.
        num_frames:
            Total number of frames available in *frames_dir*.

        Returns
        -------
        dict[int, list[tuple[int, np.ndarray]]]
            Mapping from SAM3 object ID to a list of
            ``(frame_idx, mask_uint8)`` pairs for frames where that object
            is visible.
        """
        predictor = self._get_predictor()

        # -----------------------------------------------------------------
        # Try the high-level ``handle_request`` API first (preferred,
        # available when using ``Sam3VideoPredictor``).
        # Fall back to the low-level ``init_state`` / ``propagate_in_video``
        # API exposed by ``Sam3VideoInference``.
        # -----------------------------------------------------------------
        if hasattr(predictor, "handle_request"):
            return self._run_handle_request(
                predictor, frames_dir, text_prompt, num_frames
            )
        else:
            return self._run_low_level(
                predictor, frames_dir, text_prompt, num_frames
            )

    # -------- high-level API (handle_request) --------

    def _run_handle_request(
        self,
        predictor: Any,
        frames_dir: str,
        text_prompt: str,
        num_frames: int,
    ) -> dict[int, list[tuple[int, np.ndarray]]]:
        """Use the ``handle_request`` / ``handle_stream_request`` API.

        Passes the pre-extracted *frames_dir* (a directory of JPEG images)
        as the ``resource_path`` instead of a video file.  SAM3 supports both
        video files and image directories; using a directory lets it read
        frames from disk one at a time rather than loading them all into
        GPU VRAM.

        API flow
        --------
        1. ``start_session``  → ``{"session_id": str}``
        2. ``add_prompt``     → ``{"frame_index": int, "outputs": dict}``
           where ``outputs`` is the ``obj_id_to_mask`` dict for frame 0.
        3. ``propagate_in_video`` via ``handle_stream_request``
           → generator of ``{"frame_index": int, "outputs": dict}``
        4. ``close_session``
        """
        logger.info(
            "SAM3 handle_request API: prompt='%s', frames_dir=%s (%d frames)",
            text_prompt, frames_dir, num_frames,
        )

        # 1. Start a new session pointing at the frame directory.
        #    Requesting offload_video_to_cpu so frames are read from disk
        #    one at a time — this is the key frame-by-frame memory saving.
        start_resp = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frames_dir,
                offload_video_to_cpu=True,
            )
        )
        session_id = start_resp["session_id"]

        result_map: dict[int, list[tuple[int, np.ndarray]]] = {}
        try:
            # 2. Add the text prompt at frame 0
            prompt_resp = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=text_prompt,
                )
            )
            # add_prompt returns {"frame_index": N, "outputs": obj_id_to_mask_dict}
            self._process_frame_output(prompt_resp, result_map, num_frames)

            # 3. Propagate through the rest of the video using the streaming API.
            #    handle_stream_request yields one dict per frame.
            prop_request = dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="forward",
                start_frame_index=1,         # frame 0 already handled above
                max_frame_num_to_track=num_frames,
            )
            for frame_out in predictor.handle_stream_request(request=prop_request):
                self._process_frame_output(frame_out, result_map, num_frames)

        finally:
            # 4. Always close the session to free GPU memory
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception as close_exc:
                logger.debug("close_session raised (ignored): %s", close_exc)

        return result_map

    def _process_frame_output(
        self,
        frame_out: Any,
        result_map: dict[int, list[tuple[int, np.ndarray]]],
        num_frames: int,
    ) -> None:
        """Parse one ``{"frame_index": N, "outputs": obj_id_to_mask}`` item.

        The ``outputs`` value may be either the ``obj_id_to_mask`` dict
        directly, or a wrapper dict that contains it under the key
        ``"obj_id_to_mask"``.  Both forms are handled.
        """
        if not isinstance(frame_out, dict):
            return

        frame_idx = frame_out.get("frame_index", frame_out.get("frame_idx"))
        if frame_idx is None:
            return
        if int(frame_idx) >= num_frames:
            return

        outputs = frame_out.get("outputs", {})
        # SAM3 may return outputs in two forms depending on the API version:
        #   • Wrapped:  {"obj_id_to_mask": {id: mask}, "obj_id_to_score": {id: score}}
        #   • Direct:   {id: mask}   (the obj_id_to_mask dict itself)
        # Both are normalised below.
        if isinstance(outputs, dict) and "obj_id_to_mask" in outputs:
            obj_id_to_mask = outputs["obj_id_to_mask"]
            obj_id_to_score = outputs.get("obj_id_to_score", {})
        elif isinstance(outputs, dict):
            obj_id_to_mask = outputs
            obj_id_to_score = {}
        else:
            return

        for obj_id, mask_t in obj_id_to_mask.items():
            score = float(obj_id_to_score.get(obj_id, 1.0))
            if score < self.score_threshold:
                continue
            mask_np = _mask_tensor_to_numpy(mask_t)
            result_map.setdefault(int(obj_id), []).append(
                (int(frame_idx), mask_np)
            )

    def _process_raw_outputs(
        self,
        outputs: list[Any],
        result_map: dict[int, list[tuple[int, np.ndarray]]],
        num_frames: int,
    ) -> None:
        """Parse a list of frame-output items into *result_map*.

        Each item may be a ``(frame_idx, obj_id_to_mask)`` tuple, or a dict
        in the format ``{"frame_index": N, "outputs": {...}}``.  Kept for
        backward-compat with any callers using the old batch-output style.
        """
        for item in outputs:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                frame_idx, obj_id_to_mask = item
                if int(frame_idx) >= num_frames:
                    continue
                for obj_id, mask_t in obj_id_to_mask.items():
                    mask_np = _mask_tensor_to_numpy(mask_t)
                    result_map.setdefault(int(obj_id), []).append(
                        (int(frame_idx), mask_np)
                    )
            elif isinstance(item, dict):
                self._process_frame_output(item, result_map, num_frames)

    # -------- low-level API (init_state / propagate_in_video) --------

    def _run_low_level(
        self,
        predictor: Any,
        frames_dir: str,
        text_prompt: str,
        num_frames: int,
    ) -> dict[int, list[tuple[int, np.ndarray]]]:
        """Use the ``init_state`` / ``propagate_in_video`` API directly.

        Passes the pre-extracted *frames_dir* as the resource path and
        requests ``offload_video_to_cpu=True`` so that SAM3 reads exactly
        one frame from disk into GPU VRAM at a time instead of preloading
        the entire video.
        """
        logger.info(
            "SAM3 low-level API: prompt='%s', frames_dir=%s (%d frames)",
            text_prompt, frames_dir, num_frames,
        )

        # Try with offload_video_to_cpu=True first (SAM2/SAM3 ≥ certain versions).
        # If the installed SAM3 does not support the parameter, fall back
        # gracefully to the plain call (frames still loaded from disk, just
        # not forced to CPU).
        try:
            state = predictor.init_state(
                resource_path=frames_dir,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,  # tracking state (small tensors) stays on
                                             # GPU for speed; video frames (large) are
                                             # read from disk one at a time
            )
        except TypeError:
            logger.debug(
                "SAM3 init_state() does not accept offload_video_to_cpu; "
                "falling back to plain call."
            )
            state = predictor.init_state(resource_path=frames_dir)

        # Set the text prompt in the inference state
        state["text_prompt"] = text_prompt
        try:
            state["input_batch"].find_text_batch[0] = text_prompt
        except (AttributeError, IndexError, KeyError):
            logger.debug("Could not set find_text_batch[0] directly; skipping.")

        result_map: dict[int, list[tuple[int, np.ndarray]]] = {}
        for frame_idx, out in predictor.propagate_in_video(state):
            if int(frame_idx) >= num_frames:
                break
            if out is None:
                continue
            obj_id_to_mask = out.get("obj_id_to_mask", {})
            obj_id_to_score = out.get("obj_id_to_score", {})
            for obj_id, mask_t in obj_id_to_mask.items():
                score = float(obj_id_to_score.get(obj_id, 1.0))
                if score < self.score_threshold:
                    continue
                mask_np = _mask_tensor_to_numpy(mask_t)
                result_map.setdefault(int(obj_id), []).append(
                    (int(frame_idx), mask_np)
                )

        return result_map

    # ------------------------------------------------------------------
    # Build SegmentationResult list
    # ------------------------------------------------------------------

    def _collect_results(
        self,
        num_frames: int,
        frame_size: tuple[int, int],
        player_map: dict[int, list[tuple[int, np.ndarray]]],
        ball_map: dict[int, list[tuple[int, np.ndarray]]],
        field_map: dict[int, list[tuple[int, np.ndarray]]],
    ) -> list[SegmentationResult]:
        """Assemble per-frame :class:`SegmentationResult` objects.

        Parameters
        ----------
        num_frames:
            Total number of frames (results list length).
        frame_size:
            ``(height, width)`` of the original video frames.
        player_map, ball_map, field_map:
            Per-object-ID → list-of-(frame_idx, mask) mappings returned
            by :meth:`_run_sam3_for_prompt`.
        """
        fh, fw = frame_size

        # Transpose: obj_id → frame_idx list  →  frame_idx → list-of-(obj_id, mask)
        per_frame_players: dict[int, list[tuple[int, np.ndarray]]] = {}
        for obj_id, frame_mask_list in player_map.items():
            for fidx, mask in frame_mask_list:
                per_frame_players.setdefault(fidx, []).append((obj_id, mask))

        per_frame_ball: dict[int, list[tuple[int, np.ndarray]]] = {}
        for obj_id, frame_mask_list in ball_map.items():
            for fidx, mask in frame_mask_list:
                per_frame_ball.setdefault(fidx, []).append((obj_id, mask))

        per_frame_field: dict[int, list[tuple[int, np.ndarray]]] = {}
        for obj_id, frame_mask_list in field_map.items():
            for fidx, mask in frame_mask_list:
                per_frame_field.setdefault(fidx, []).append((obj_id, mask))

        results: list[SegmentationResult] = []
        for frame_idx in range(num_frames):
            seg = SegmentationResult(frame_index=frame_idx)

            # ----- Players -----
            for obj_id, mask in per_frame_players.get(frame_idx, []):
                mask_rs = self._ensure_shape(mask, fh, fw)
                bbox = _mask_to_bbox(mask_rs)
                if bbox is None:
                    continue
                seg.player_ids.append(obj_id)
                seg.player_masks.append(mask_rs)
                seg.player_bboxes.append(bbox)

            # ----- Ball -----
            ball_entries = per_frame_ball.get(frame_idx, [])
            if ball_entries:
                # Use the largest-area ball mask
                best_obj_id, best_mask = max(
                    ball_entries,
                    key=lambda t: int(np.count_nonzero(t[1])),
                )
                ball_mask = self._ensure_shape(best_mask, fh, fw)
                ball_bbox = _mask_to_bbox(ball_mask)
                if ball_bbox is not None:
                    seg.ball_mask = ball_mask
                    seg.ball_bbox = ball_bbox
                    x1, y1, x2, y2 = ball_bbox
                    seg.ball_center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
                    seg.ball_source = "detected"

            # ----- Field -----
            field_entries = per_frame_field.get(frame_idx, [])
            if field_entries:
                # Merge all field-segment masks (there is usually only one)
                field_masks = [
                    self._ensure_shape(m, fh, fw) for _, m in field_entries
                ]
                union = _union_masks(field_masks, (fh, fw))
                field_bbox = _mask_to_bbox(union)
                if field_bbox is not None:
                    seg.field_mask = union
                    seg.field_bbox = field_bbox

            results.append(seg)

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_frames_to_dir(
        self,
        video_path: str,
        frames_dir: str,
        max_frames: int | None,
    ) -> tuple[int, int, int]:
        """Decode video frames to JPEG files in *frames_dir*.

        Files are named ``000000.jpg``, ``000001.jpg``, … so SAM3 reads them
        in the correct order.  Only the current frame is in CPU memory at any
        one time — no GPU is involved.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        frames_dir:
            Directory where JPEG frame images will be written.
        max_frames:
            If set, stop after writing this many frames.

        Returns
        -------
        tuple[int, int, int]
            ``(num_frames, frame_height, frame_width)``.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.frame_jpeg_quality]

        idx = 0
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            out_path = os.path.join(frames_dir, f"{idx:06d}.jpg")
            cv2.imwrite(out_path, frame, encode_params)
            idx += 1
            if idx % 500 == 0:
                logger.debug("Frame extraction: %d frames written...", idx)

        cap.release()
        logger.info(
            "Extracted %d frames (%dx%d) from '%s' → '%s'",
            idx, fw, fh, video_path, frames_dir,
        )
        return idx, fh, fw

    def process_video(
        self,
        video_path: str,
        max_frames: int | None = None,
    ) -> list[SegmentationResult]:
        """Process a video using SAM3 text-prompt segmentation.

        Frame-by-frame memory model
        ----------------------------
        Frames are first extracted from the video into a temporary directory
        using OpenCV (CPU-only, no GPU involved).  SAM3 is then pointed at
        that directory and reads frames from disk one at a time during
        inference, so GPU VRAM usage is proportional to the model size rather
        than the video length.

        The temporary directory is automatically cleaned up when inference
        completes (or if an error occurs).

        Runs up to three SAM3 inference passes (players, ball, field) then
        merges the results into a :class:`SegmentationResult` per frame.

        Parameters
        ----------
        video_path:
            Path to the input video file.
        max_frames:
            If set, process at most this many frames.

        Returns
        -------
        list[SegmentationResult]
            One result per processed frame, in order.
        """
        logger.info("SAM3: extracting frames from '%s'…", video_path)

        with tempfile.TemporaryDirectory(prefix="sam3_frames_") as frames_dir:
            num_frames, fh, fw = self._extract_frames_to_dir(
                video_path, frames_dir, max_frames
            )

            if num_frames == 0:
                raise ValueError(f"No frames could be read from video: {video_path}")

            logger.info(
                "SAM3 processing: %d frames (%dx%d) in '%s'",
                num_frames, fw, fh, frames_dir,
            )

            # ----- Player segmentation pass -----
            logger.info(
                "=== SAM3 player pass: prompt='%s' ===", self.player_text_prompt
            )
            player_map = self._run_sam3_for_prompt(
                frames_dir, self.player_text_prompt, num_frames
            )
            logger.info(
                "SAM3 player pass: %d unique objects detected", len(player_map)
            )

            # ----- Ball detection pass -----
            ball_map: dict[int, list[tuple[int, np.ndarray]]] = {}
            if self.ball_text_prompt:
                logger.info(
                    "=== SAM3 ball pass: prompt='%s' ===", self.ball_text_prompt
                )
                ball_map = self._run_sam3_for_prompt(
                    frames_dir, self.ball_text_prompt, num_frames
                )
                logger.info(
                    "SAM3 ball pass: %d unique objects detected", len(ball_map)
                )

            # ----- Field segmentation pass -----
            field_map: dict[int, list[tuple[int, np.ndarray]]] = {}
            if self.field_text_prompt:
                logger.info(
                    "=== SAM3 field pass: prompt='%s' ===", self.field_text_prompt
                )
                field_map = self._run_sam3_for_prompt(
                    frames_dir, self.field_text_prompt, num_frames
                )
                logger.info(
                    "SAM3 field pass: %d unique objects detected", len(field_map)
                )

            # ----- Assemble results -----
            results = self._collect_results(
                num_frames, (fh, fw), player_map, ball_map, field_map
            )

        # frames_dir is automatically removed here by the context manager
        logger.info("SAM3 processing complete: %d frames", len(results))
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_shape(
        mask: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> np.ndarray:
        """Resize *mask* to ``(target_h, target_w)`` if needed."""
        mh, mw = mask.shape[:2]
        if mh != target_h or mw != target_w:
            mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
        return mask
