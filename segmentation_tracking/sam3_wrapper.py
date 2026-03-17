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
from dataclasses import dataclass, field as dc_field
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
# Internal per-session state
# ---------------------------------------------------------------------------

@dataclass
class _Sam3Session:
    """Holds everything needed for one SAM3 inference session."""

    predictor: Any                  # Sam3VideoInference or high-level predictor
    inference_state: Any            # dict returned by predictor.init_state(...)
    session_id: Any = None          # used when the high-level handle_request API is used
    api_style: str = "low_level"    # "low_level" | "handle_request"
    num_frames: int = 0

    # Collected per-frame results: list of (frame_idx, obj_id_to_mask)
    frame_results: list[tuple[int, dict[int, np.ndarray]]] = dc_field(
        default_factory=list
    )


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
    """

    def __init__(
        self,
        sam3_model_path: str = _DEFAULT_MODEL_PATH,
        device: str = "cuda",
        player_text_prompt: str = _DEFAULT_PLAYER_PROMPT,
        ball_text_prompt: str | None = _DEFAULT_BALL_PROMPT,
        field_text_prompt: str | None = _DEFAULT_FIELD_PROMPT,
        score_threshold: float = _DEFAULT_SCORE_THRESH,
    ) -> None:
        self.sam3_model_path = sam3_model_path
        self.device = device
        self.player_text_prompt = player_text_prompt
        self.ball_text_prompt = ball_text_prompt
        self.field_text_prompt = field_text_prompt
        self.score_threshold = score_threshold

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
            from sam3.model_builder import build_sam3_video_predictor
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
            checkpoint = None
        else:
            checkpoint = self.sam3_model_path

        logger.info("Loading SAM3 video predictor (checkpoint=%s)", checkpoint)
        self._predictor = build_sam3_video_predictor(
            checkpoint=checkpoint,
            device=self.device,
        )
        return self._predictor

    # ------------------------------------------------------------------
    # Single-category SAM3 inference pass
    # ------------------------------------------------------------------

    def _run_sam3_for_prompt(
        self,
        video_path: str,
        text_prompt: str,
        max_frames: int | None,
    ) -> dict[int, list[tuple[int, np.ndarray]]]:
        """Run SAM3 for one text prompt and collect per-frame masks.

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
        # available when using ``build_sam3_video_predictor``).
        # Fall back to the low-level ``init_state`` / ``propagate_in_video``
        # API exposed by ``Sam3VideoInference``.
        # -----------------------------------------------------------------
        if hasattr(predictor, "handle_request"):
            return self._run_handle_request(
                predictor, video_path, text_prompt, max_frames
            )
        else:
            return self._run_low_level(
                predictor, video_path, text_prompt, max_frames
            )

    # -------- high-level API (handle_request) --------

    def _run_handle_request(
        self,
        predictor: Any,
        video_path: str,
        text_prompt: str,
        max_frames: int | None,
    ) -> dict[int, list[tuple[int, np.ndarray]]]:
        """Use the ``handle_request`` request/response API."""
        logger.info(
            "SAM3 handle_request API: prompt='%s', video=%s", text_prompt, video_path
        )

        # Start a new session
        start_resp = predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = start_resp["session_id"]

        # Add the text prompt at frame 0 — SAM3 propagates automatically
        prompt_resp = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=text_prompt,
            )
        )

        # The response may already contain per-frame outputs
        raw_outputs: list[Any] = prompt_resp.get("outputs", [])

        # Additionally pull frame-by-frame results when available
        result_map: dict[int, list[tuple[int, np.ndarray]]] = {}
        self._process_raw_outputs(raw_outputs, result_map, max_frames)

        # If the response bundles all frames, we're done; otherwise run propagation
        if not result_map:
            logger.debug("No outputs in add_prompt response; running propagation.")
            prop_resp = predictor.handle_request(
                request=dict(
                    type="propagate",
                    session_id=session_id,
                    max_frames=max_frames,
                )
            )
            self._process_raw_outputs(
                prop_resp.get("outputs", []), result_map, max_frames
            )

        return result_map

    def _process_raw_outputs(
        self,
        outputs: list[Any],
        result_map: dict[int, list[tuple[int, np.ndarray]]],
        max_frames: int | None,
    ) -> None:
        """Parse a list of ``(frame_idx, obj_id_to_mask)`` pairs into *result_map*."""
        for item in outputs:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                frame_idx, obj_id_to_mask = item
            elif isinstance(item, dict):
                frame_idx = item.get("frame_idx", item.get("frame_index", None))
                obj_id_to_mask = item.get("obj_id_to_mask", {})
            else:
                continue

            if frame_idx is None:
                continue
            if max_frames is not None and int(frame_idx) >= max_frames:
                continue

            for obj_id, mask_t in obj_id_to_mask.items():
                mask_np = _mask_tensor_to_numpy(mask_t)
                result_map.setdefault(int(obj_id), []).append(
                    (int(frame_idx), mask_np)
                )

    # -------- low-level API (init_state / propagate_in_video) --------

    def _run_low_level(
        self,
        predictor: Any,
        video_path: str,
        text_prompt: str,
        max_frames: int | None,
    ) -> dict[int, list[tuple[int, np.ndarray]]]:
        """Use the ``init_state`` / ``propagate_in_video`` API directly."""
        logger.info(
            "SAM3 low-level API: prompt='%s', video=%s", text_prompt, video_path
        )

        state = predictor.init_state(resource_path=video_path)

        # Set the text prompt in the inference state
        state["text_prompt"] = text_prompt
        try:
            state["input_batch"].find_text_batch[0] = text_prompt
        except (AttributeError, IndexError, KeyError):
            logger.debug("Could not set find_text_batch[0] directly; skipping.")

        result_map: dict[int, list[tuple[int, np.ndarray]]] = {}
        for frame_idx, out in predictor.propagate_in_video(state):
            if max_frames is not None and int(frame_idx) >= max_frames:
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

    def process_video(
        self,
        video_path: str,
        max_frames: int | None = None,
    ) -> list[SegmentationResult]:
        """Process a video using SAM3 text-prompt segmentation.

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
        # Read video metadata
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        if max_frames:
            total_frames = min(total_frames, max_frames)
        logger.info(
            "SAM3 processing: video=%s (%d frames, %dx%d)",
            video_path, total_frames, fw, fh,
        )

        # ----- Player segmentation pass -----
        logger.info("=== SAM3 player pass: prompt='%s' ===", self.player_text_prompt)
        player_map = self._run_sam3_for_prompt(
            video_path, self.player_text_prompt, max_frames
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
                video_path, self.ball_text_prompt, max_frames
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
                video_path, self.field_text_prompt, max_frames
            )
            logger.info(
                "SAM3 field pass: %d unique objects detected", len(field_map)
            )

        # ----- Assemble results -----
        results = self._collect_results(
            total_frames, (fh, fw), player_map, ball_map, field_map
        )
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
