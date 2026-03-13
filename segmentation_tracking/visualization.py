"""
visualization.py
----------------
Frame annotation: semi-transparent segmentation masks, player IDs, pose
skeletons, ball overlay, and velocity vector-field attractor.

Public API
~~~~~~~~~~
``Visualizer``
    Stateless helper class.  Call ``Visualizer.draw_frame(frame, player_tracks,
    ball_track)`` to get an annotated copy of a frame.
"""

from __future__ import annotations

import colorsys
import logging
from typing import Sequence

import cv2
import numpy as np

from segmentation_tracking.association import (
    BallTrack,
    COCO_SKELETON,
    PlayerTrack,
)
from segmentation_tracking.vector_field import AttractorEstimate

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_MASK_ALPHA = 0.40          # transparency of the segmentation overlay
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.6
_LABEL_THICKNESS = 2
_SKEL_THICKNESS = 2
_KP_RADIUS = 4
_KP_CONF_THRESHOLD = 0.3    # draw keypoints only above this confidence
_BALL_COLOR = (0, 255, 255)  # cyan for ball (default / no debug)
_BALL_MASK_ALPHA = 0.50

# Ball detection source colours (BGR) — used in debug mode
_BALL_SOURCE_COLORS: dict[str, tuple[int, int, int]] = {
    "detected":   (0, 220, 0),    # bright green  – confirmed YOLO detection
    "roi":        (0, 200, 100),  # yellow-green   – ROI YOLO re-detection (FRoG-MOT)
    "mosse":      (0, 165, 255),  # orange         – MOSSE correlation gap-fill
    "cotracker":  (255, 100, 0),  # blue           – CoTracker3 propagation
    "predicted":  (0, 0, 220),    # red            – velocity extrapolation only
    "none":       (200, 200, 200),# grey           – fallback (should not appear)
}

# Human-readable labels for each source
_BALL_SOURCE_LABELS: dict[str, str] = {
    "detected":  "DETECT",
    "roi":       "ROI-DET",
    "mosse":     "MOSSE",
    "cotracker": "CT3",
    "predicted": "PRED",
    "none":      "NONE",
}

# Golden ratio conjugate – spreads player IDs to perceptually distinct hues
_GOLDEN_RATIO = 0.6180339887

# Attractor visualisation constants
_ATTRACTOR_COLOR    = (0, 255, 255)    # cyan diamond (same as default ball, but distinct marker)
_ATTRACTOR_HI_COLOR = (0, 255, 200)    # high-confidence → bright cyan-green
_ATTRACTOR_LO_COLOR = (0, 130, 255)    # low-confidence  → orange-yellow
_VECTOR_ALPHA       = 0.70             # opacity of velocity arrows


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def _id_to_color(player_id: int) -> tuple[int, int, int]:
    """Map a player ID to a distinct BGR colour using the golden-ratio hue."""
    hue = (player_id * _GOLDEN_RATIO) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class Visualizer:
    """Annotates video frames with masks, player IDs, and pose skeletons.

    Parameters
    ----------
    mask_alpha:
        Opacity of the segmentation colour overlay (0 = transparent,
        1 = opaque).  Defaults to 0.40.
    show_bbox:
        Draw the bounding box rectangle around each player.
    show_id:
        Render the ``[Player ID: N]`` label above the bounding box.
    show_skeleton:
        Render the COCO pose skeleton.
    show_ball:
        Render the ball mask overlay and centre dot.
    show_ball_debug:
        When *True*, colour-code the ball marker by detection source and
        draw a cumulative detection-statistics HUD in the top-right corner.
        Sources are colour-coded as follows:

        * Green   → YOLO detection (stage-1).
        * Yellow-green → ROI YOLO re-detection (FRoG-MOT stage-2).
        * Orange  → MOSSE correlation filter gap-fill.
        * Red     → velocity extrapolation only.

        Defaults to *False*.
    show_attractor:
        When *True*, draw per-player velocity arrows and the vector-field
        **attractor** — the point that the player velocity rays collectively
        converge towards, which estimates the ball position.  The attractor
        is shown as a rotating diamond marker with a confidence-coloured
        ring:

        * Bright cyan-green → high confidence (many fast players pointing to it).
        * Orange-yellow     → low confidence (few players or ambiguous directions).

        Requires caller to pass ``velocities`` and ``attractor`` to
        :meth:`draw_frame`.  Defaults to *False*.
    show_ct_trajectories:
        When *True*, draw the per-player CoTracker3 point trajectory trails.
        Each trail is a polyline of the last *ct_trail_len* positions for
        each tracked keypoint cluster, colour-coded by player ID.  Requires
        the caller to pass ``ct_trajectories`` to :meth:`draw_frame`.
        Defaults to *False*.
    keypoint_conf_threshold:
        Only draw keypoints whose confidence exceeds this value.
    """

    def __init__(
        self,
        mask_alpha: float = _MASK_ALPHA,
        show_bbox: bool = True,
        show_id: bool = True,
        show_skeleton: bool = True,
        show_ball: bool = True,
        show_ball_debug: bool = False,
        show_attractor: bool = False,
        show_ct_trajectories: bool = False,
        keypoint_conf_threshold: float = _KP_CONF_THRESHOLD,
    ) -> None:
        self.mask_alpha = mask_alpha
        self.show_bbox = show_bbox
        self.show_id = show_id
        self.show_skeleton = show_skeleton
        self.show_ball = show_ball
        self.show_ball_debug = show_ball_debug
        self.show_attractor = show_attractor
        self.show_ct_trajectories = show_ct_trajectories
        self.keypoint_conf_threshold = keypoint_conf_threshold

        # Running detection-source counters (reset on demand via reset_ball_stats)
        self._ball_counts: dict[str, int] = {
            "detected": 0,
            "roi": 0,
            "mosse": 0,
            "cotracker": 0,
            "predicted": 0,
            "none": 0,
        }
        self._total_frames: int = 0

    # ── Public ────────────────────────────────────────────────────────────────

    def reset_ball_stats(self) -> None:
        """Reset the cumulative ball detection counters."""
        self._ball_counts = {k: 0 for k in self._ball_counts}
        self._total_frames = 0

    def draw_frame(
        self,
        frame: np.ndarray,
        player_tracks: Sequence[PlayerTrack],
        ball_track: BallTrack | None = None,
        frame_idx: int | None = None,
        velocities: dict[int, tuple[float, float, float, float]] | None = None,
        attractor: AttractorEstimate | None = None,
        ct_trajectories: dict[int, list[tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        """
        Annotate a single frame and return the result.

        The original *frame* is not modified.

        Parameters
        ----------
        frame:
            BGR image array, shape ``(H, W, 3)``.
        player_tracks:
            Iterable of :class:`~segmentation_tracking.association.PlayerTrack`
            for the current frame.
        ball_track:
            Optional :class:`~segmentation_tracking.association.BallTrack`.
        frame_idx:
            Optional 0-based frame index; shown in the debug HUD when
            ``show_ball_debug`` is enabled.
        velocities:
            Optional mapping ``{player_id: (cx, cy, vx, vy)}`` from
            :class:`~segmentation_tracking.vector_field.PlayerVelocityTracker`.
            Used when ``show_attractor`` is *True* to draw velocity arrows.
        attractor:
            Optional :class:`~segmentation_tracking.vector_field.AttractorEstimate`
            from :func:`~segmentation_tracking.vector_field.estimate_attractor`.
            Drawn when ``show_attractor`` is *True*.
        ct_trajectories:
            Optional mapping ``{player_id: [(x0,y0), ..., (x_cur,y_cur)]}``
            from
            :meth:`~segmentation_tracking.vector_field.KeypointVelocityTracker.get_ct_trajectories`.
            Drawn when ``show_ct_trajectories`` is *True*.

        Returns
        -------
        np.ndarray
            Annotated BGR image of the same shape as *frame*.
        """
        canvas = frame.copy()

        # Draw CoTracker3 trajectories beneath everything else
        if self.show_ct_trajectories and ct_trajectories:
            self._draw_ct_trajectories(canvas, ct_trajectories)

        # Draw player segmentation masks (colour overlay)
        for track in player_tracks:
            color = _id_to_color(track.id)
            canvas = self._draw_mask(canvas, track.mask, color)

        # Draw velocity arrows behind bounding boxes (so boxes are on top)
        if self.show_attractor and velocities:
            self._draw_velocity_vectors(canvas, velocities)

        # Draw player bounding boxes, IDs, and skeletons on top of masks
        for track in player_tracks:
            color = _id_to_color(track.id)
            if self.show_bbox:
                self._draw_bbox(canvas, track.bbox, color)
            if self.show_id:
                self._draw_label(canvas, track.id, track.bbox, color, track.team_label)
            if self.show_skeleton and track.keypoints is not None:
                self._draw_skeleton(canvas, track.keypoints, track.keypoint_scores, color)

        # Draw ball
        if self.show_ball and ball_track is not None:
            canvas = self._draw_ball(canvas, ball_track)

        # Draw vector-field attractor
        if self.show_attractor and attractor is not None:
            self._draw_attractor(canvas, attractor)

        # Update running counters and draw debug HUD
        self._total_frames += 1
        source = ball_track.source if ball_track is not None else "none"
        if source in self._ball_counts:
            self._ball_counts[source] += 1
        else:
            self._ball_counts["none"] += 1

        if self.show_ball_debug:
            canvas = self._draw_ball_stats_hud(canvas, frame_idx)

        return canvas

    # ── Private drawing helpers ───────────────────────────────────────────────

    def _draw_mask(
        self,
        canvas: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        """Blend a semi-transparent coloured mask onto *canvas*."""
        if mask is None or not mask.any():
            return canvas

        h, w = canvas.shape[:2]
        mh, mw = mask.shape
        if mh != h or mw != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        color_layer = np.zeros_like(canvas, dtype=np.uint8)
        color_layer[mask > 0] = color

        # alpha-blend only in the mask region
        mask_bool = mask > 0
        canvas[mask_bool] = cv2.addWeighted(
            canvas, 1.0 - self.mask_alpha,
            color_layer, self.mask_alpha,
            0,
        )[mask_bool]

        return canvas

    def _draw_ct_trajectories(
        self,
        canvas: np.ndarray,
        ct_trajectories: dict[int, list[tuple[float, float]]],
    ) -> None:
        """Draw CoTracker3 point trajectory trails as colour-coded polylines.

        Each player's trail is drawn as a series of connected circles and
        line segments fading from semi-transparent (oldest) to opaque (newest).
        The colour matches the player's ID colour so trails are visually
        associated with the correct player.

        Parameters
        ----------
        canvas:
            BGR image to draw onto (modified in-place).
        ct_trajectories:
            ``{player_id: [(x0,y0), ..., (x_cur,y_cur)]}`` mapping, oldest
            position first.
        """
        for pid, pts in ct_trajectories.items():
            if len(pts) < 2:
                continue
            color = _id_to_color(pid)
            n = len(pts)
            for i in range(1, n):
                # Fade opacity: oldest segment is 20% opacity, newest is 100%
                alpha = 0.20 + 0.80 * (i / (n - 1))
                x0, y0 = int(round(pts[i - 1][0])), int(round(pts[i - 1][1]))
                x1, y1 = int(round(pts[i][0])), int(round(pts[i][1]))
                # Blend the line segment with the canvas
                overlay = canvas.copy()
                cv2.line(overlay, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
            # Draw a solid dot at the current (newest) position
            cx, cy = int(round(pts[-1][0])), int(round(pts[-1][1]))
            cv2.circle(canvas, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_bbox(
        canvas: np.ndarray,
        bbox: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    @staticmethod
    def _draw_label(
        canvas: np.ndarray,
        player_id: int,
        bbox: np.ndarray,
        color: tuple[int, int, int],
        team_label: int | None = None,
    ) -> None:
        team_suffix = f" T{team_label}" if team_label is not None else ""
        label = f"P{player_id}{team_suffix}"
        x1, y1 = int(bbox[0]), int(bbox[1])

        (tw, th), baseline = cv2.getTextSize(
            label, _LABEL_FONT, _LABEL_SCALE, _LABEL_THICKNESS
        )
        # Position label above the bbox; clamp to frame top
        lx, ly = x1, max(y1 - baseline - 4, th + baseline)

        # Background rectangle
        cv2.rectangle(
            canvas,
            (lx - 2, ly - th - baseline - 2),
            (lx + tw + 2, ly + baseline + 2),
            color,
            cv2.FILLED,
        )
        # Text in white for contrast
        cv2.putText(
            canvas, label, (lx, ly),
            _LABEL_FONT, _LABEL_SCALE, (255, 255, 255), _LABEL_THICKNESS,
            cv2.LINE_AA,
        )

    def _draw_skeleton(
        self,
        canvas: np.ndarray,
        keypoints: np.ndarray,
        keypoint_scores: np.ndarray | None,
        color: tuple[int, int, int],
    ) -> None:
        """Draw COCO skeleton lines and keypoint circles."""
        scores = (
            keypoint_scores
            if keypoint_scores is not None
            else np.ones(len(keypoints), dtype=np.float32)
        )
        # Draw bones
        for i, j in COCO_SKELETON:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            if scores[i] < self.keypoint_conf_threshold:
                continue
            if scores[j] < self.keypoint_conf_threshold:
                continue
            xi, yi = int(keypoints[i][0]), int(keypoints[i][1])
            xj, yj = int(keypoints[j][0]), int(keypoints[j][1])
            if xi == 0 and yi == 0:
                continue
            if xj == 0 and yj == 0:
                continue
            cv2.line(canvas, (xi, yi), (xj, yj), color, _SKEL_THICKNESS, cv2.LINE_AA)

        # Draw joints
        for idx, (kp, sc) in enumerate(zip(keypoints, scores)):
            if sc < self.keypoint_conf_threshold:
                continue
            x, y = int(kp[0]), int(kp[1])
            if x == 0 and y == 0:
                continue
            cv2.circle(canvas, (x, y), _KP_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), _KP_RADIUS, color, 1, cv2.LINE_AA)

    def _draw_ball(
        self,
        canvas: np.ndarray,
        ball_track: BallTrack,
    ) -> np.ndarray:
        """Draw ball mask overlay and centre marker.

        When ``show_ball_debug`` is *True* the ball circle and label are
        colour-coded by detection source so the viewer can immediately tell
        whether the position came from a YOLO detection, MOSSE gap-fill, or
        pure velocity extrapolation.
        """
        source = getattr(ball_track, "source", "none")
        if self.show_ball_debug:
            ball_color = _BALL_SOURCE_COLORS.get(source, _BALL_COLOR)
        else:
            ball_color = _BALL_COLOR

        if ball_track.mask is not None and ball_track.mask.any():
            h, w = canvas.shape[:2]
            bm = ball_track.mask
            if bm.shape[0] != h or bm.shape[1] != w:
                bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST)

            color_layer = np.zeros_like(canvas, dtype=np.uint8)
            color_layer[bm > 0] = ball_color

            mask_bool = bm > 0
            canvas[mask_bool] = cv2.addWeighted(
                canvas, 1.0 - _BALL_MASK_ALPHA,
                color_layer, _BALL_MASK_ALPHA,
                0,
            )[mask_bool]

        if ball_track.bbox is not None:
            x1, y1, x2, y2 = ball_track.bbox.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), ball_color, 2)

        # Centre dot
        cx, cy = int(ball_track.center[0]), int(ball_track.center[1])
        cv2.circle(canvas, (cx, cy), 6, ball_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 6, (0, 0, 0), 1, cv2.LINE_AA)

        # Label — include source tag when debug mode is active
        if self.show_ball_debug:
            src_tag = _BALL_SOURCE_LABELS.get(source, source.upper())
            label = f"Ball [{src_tag}]"
        else:
            label = "Ball"

        cv2.putText(
            canvas, label,
            (cx + 8, cy - 8),
            _LABEL_FONT, _LABEL_SCALE, ball_color, _LABEL_THICKNESS, cv2.LINE_AA,
        )

        return canvas

    def _draw_ball_stats_hud(
        self,
        canvas: np.ndarray,
        frame_idx: int | None = None,
    ) -> np.ndarray:
        """Render a semi-transparent ball detection statistics panel.

        Shows cumulative counts for each ball position source
        (DETECT / ROI-DET / MOSSE / PRED / NONE) and the overall
        detection rate.  Drawn in the top-right corner of the frame.

        This is intentionally lightweight — pure OpenCV drawing, no
        external dependencies.
        """
        h, w = canvas.shape[:2]

        total = self._total_frames or 1
        det_n  = self._ball_counts.get("detected", 0)
        roi_n  = self._ball_counts.get("roi", 0)
        mos_n  = self._ball_counts.get("mosse", 0)
        pred_n = self._ball_counts.get("predicted", 0)
        none_n = self._ball_counts.get("none", 0)
        any_det = det_n + roi_n       # frames with a real detection
        det_rate = 100.0 * any_det / total

        # Build lines
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if frame_idx is not None:
            lines.append((f"Frame {frame_idx}", (220, 220, 220)))
        lines.append((f"Frames total : {total}", (220, 220, 220)))
        lines.append((
            f"DETECT  : {det_n:4d}  ({100.0*det_n/total:4.1f}%)",
            _BALL_SOURCE_COLORS["detected"],
        ))
        lines.append((
            f"ROI-DET : {roi_n:4d}  ({100.0*roi_n/total:4.1f}%)",
            _BALL_SOURCE_COLORS["roi"],
        ))
        lines.append((
            f"MOSSE   : {mos_n:4d}  ({100.0*mos_n/total:4.1f}%)",
            _BALL_SOURCE_COLORS["mosse"],
        ))
        lines.append((
            f"PRED    : {pred_n:4d}  ({100.0*pred_n/total:4.1f}%)",
            _BALL_SOURCE_COLORS["predicted"],
        ))
        lines.append((
            f"NONE    : {none_n:4d}  ({100.0*none_n/total:4.1f}%)",
            (180, 180, 180),
        ))
        lines.append((
            f"Det rate: {det_rate:5.1f}%",
            (0, 220, 0) if det_rate >= 50 else (0, 100, 255),
        ))

        font = _LABEL_FONT
        fscale = 0.48
        fthick = 1
        pad = 6

        # Measure max width
        max_tw = 0
        line_h = 0
        for txt, _ in lines:
            (tw, th), bl = cv2.getTextSize(txt, font, fscale, fthick)
            max_tw = max(max_tw, tw)
            line_h = max(line_h, th + bl)

        panel_w = max_tw + 2 * pad
        panel_h = len(lines) * (line_h + 4) + 2 * pad

        # Position: top-right corner
        x0 = w - panel_w - 4
        y0 = 4

        # Semi-transparent dark background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), cv2.FILLED)
        # Blend: overlay (dark rect) 65% + original canvas 35% → blended result
        blended = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0)
        canvas[:] = blended

        # Draw each line
        for i, (txt, color) in enumerate(lines):
            ty = y0 + pad + (i + 1) * (line_h + 4) - 2
            cv2.putText(canvas, txt, (x0 + pad, ty), font, fscale, color, fthick, cv2.LINE_AA)

        return canvas

    # ── Vector-field attractor helpers ────────────────────────────────────────

    def _draw_velocity_vectors(
        self,
        canvas: np.ndarray,
        velocities: dict[int, tuple[float, float, float, float]],
        scale: float = 8.0,
    ) -> None:
        """Draw semi-transparent velocity arrows from each player's centre.

        Each arrow is colour-coded by the player's ID (same palette as the
        bounding boxes) and scaled proportionally to their speed.  Arrows are
        drawn on a separate layer and blended at *_VECTOR_ALPHA* opacity so
        they don't obscure other annotations.

        Parameters
        ----------
        canvas:
            BGR frame to annotate in-place.
        velocities:
            ``{player_id: (cx, cy, vx, vy)}`` from
            :class:`~segmentation_tracking.vector_field.PlayerVelocityTracker`.
        scale:
            Multiplier that converts velocity in px/frame to displayed arrow
            length in pixels.  Defaults to 8 (so a 10 px/frame speed gives
            an 80-pixel arrow).
        """
        if not velocities:
            return

        overlay = canvas.copy()

        for pid, (cx, cy, vx, vy) in velocities.items():
            speed = float(np.hypot(vx, vy))
            if speed < 0.1:
                continue

            color = _id_to_color(pid)
            ox, oy = int(cx), int(cy)
            tx = int(cx + vx * scale)
            ty = int(cy + vy * scale)

            # Arrow shaft
            cv2.arrowedLine(
                overlay, (ox, oy), (tx, ty),
                color, 2, cv2.LINE_AA, tipLength=0.35,
            )
            # Speed label at tip
            spd_txt = f"{speed:.1f}"
            cv2.putText(
                overlay, spd_txt, (tx + 3, ty - 3),
                _LABEL_FONT, 0.38, color, 1, cv2.LINE_AA,
            )

        # Blend arrows onto canvas
        blended = cv2.addWeighted(overlay, _VECTOR_ALPHA, canvas, 1.0 - _VECTOR_ALPHA, 0)
        canvas[:] = blended

    def _draw_attractor(
        self,
        canvas: np.ndarray,
        attractor: AttractorEstimate,
    ) -> None:
        """Draw the vector-field attractor as a diamond marker with a label.

        The marker colour interpolates between :data:`_ATTRACTOR_LO_COLOR`
        (low confidence) and :data:`_ATTRACTOR_HI_COLOR` (high confidence)
        based on ``attractor.confidence``.

        When ``attractor.is_held`` is *True* (the Kalman smoother is predicting
        from a stale estimate), the diamond outline is dashed and the label
        carries a ``[HELD]`` tag and the stale-frame count.  The marker
        becomes progressively more transparent as ``stale_frames`` grows.

        Parameters
        ----------
        canvas:
            BGR frame to annotate in-place.
        attractor:
            :class:`~segmentation_tracking.vector_field.AttractorEstimate`.
        """
        ax, ay = int(attractor.point[0]), int(attractor.point[1])
        conf = float(attractor.confidence)
        is_held = getattr(attractor, "is_held", False)
        stale_frames = getattr(attractor, "stale_frames", 0)

        # Interpolate colour between lo (orange) and hi (cyan-green)
        lo = np.array(_ATTRACTOR_LO_COLOR, dtype=np.float32)
        hi = np.array(_ATTRACTOR_HI_COLOR, dtype=np.float32)
        color_f = lo + conf * (hi - lo)
        color = (int(color_f[0]), int(color_f[1]), int(color_f[2]))

        # Diamond half-size (grows slightly with confidence, shrinks when held)
        r = max(6, int(10 + conf * 8))
        if is_held:
            r = max(6, int(r * 0.8))

        # Diamond vertices: top, right, bottom, left
        pts = np.array([
            [ax,     ay - r],
            [ax + r, ay    ],
            [ax,     ay + r],
            [ax - r, ay    ],
        ], dtype=np.int32)

        # Fill opacity: full (0.55) when fresh; fades linearly toward 0.15
        # when held, using the already-decayed confidence as the fade proxy
        # so the visual transparency matches the semantic confidence.
        _FILL_ALPHA_MAX = 0.55
        _FILL_ALPHA_MIN = 0.15
        if not is_held:
            fill_alpha = _FILL_ALPHA_MAX
        else:
            # conf is already 0→ base when stale; re-map to [_MIN, _MAX] range
            fill_alpha = _FILL_ALPHA_MIN + conf * (_FILL_ALPHA_MAX - _FILL_ALPHA_MIN)

        # Semi-transparent fill on an overlay
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color)
        blended = cv2.addWeighted(overlay, fill_alpha, canvas, 1.0 - fill_alpha, 0)
        canvas[:] = blended

        # Outline — solid for fresh estimate, dashed-style (dotted segments) for held
        if not is_held:
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        else:
            # Draw "dashed" diamond by segmenting each edge into alternating pieces
            verts = [pts[0], pts[1], pts[2], pts[3], pts[0]]
            for vi in range(len(verts) - 1):
                p0 = verts[vi]
                p1 = verts[vi + 1]
                # Draw only first 60% of each edge (creates a dashed look)
                mid = (int(0.4 * p0[0] + 0.6 * p1[0]), int(0.4 * p0[1] + 0.6 * p1[1]))
                cv2.line(canvas, tuple(p0), mid, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.line(canvas, tuple(p0), mid, color, 1, cv2.LINE_AA)

        # Centre dot
        cv2.circle(canvas, (ax, ay), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Label
        held_tag = f" [HELD:{stale_frames}f]" if is_held else ""
        label1 = f"Attractor (conf={conf:.2f}){held_tag}"
        label2 = f"n={attractor.n_players}  spd={attractor.mean_speed:.1f}px/f"
        lx = ax + r + 6
        ly = ay - 4
        for label in (label1, label2):
            (tw, th), bl = cv2.getTextSize(label, _LABEL_FONT, 0.50, 1)
            cv2.rectangle(
                canvas,
                (lx - 2, ly - th - 2),
                (lx + tw + 2, ly + bl + 2),
                (20, 20, 20),
                cv2.FILLED,
            )
            cv2.putText(
                canvas, label, (lx, ly),
                _LABEL_FONT, 0.50, color, 1, cv2.LINE_AA,
            )
            ly += th + bl + 6
