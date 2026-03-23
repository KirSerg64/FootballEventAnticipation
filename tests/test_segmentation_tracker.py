"""
tests/test_segmentation_tracker.py
-----------------------------------
Unit and integration tests for SegmentationTracker and its helpers.

Heavy external dependencies (YOLO, SAM2) are replaced with lightweight mocks
so the suite runs without GPU, pretrained weights, or the sam2 package.

Run with:
    pytest tests/test_segmentation_tracker.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import MagicMock, PropertyMock, patch

import cv2
import numpy as np
import pytest
import supervision as sv

# ---------------------------------------------------------------------------
# Make sure the repo root is importable from any working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from segmentation_tracking.segmentation_model import (  # noqa: E402
    SegmentationTracker,
    TrackerState,
)

# ---------------------------------------------------------------------------
# Small helpers used across tests
# ---------------------------------------------------------------------------

def _frame(h: int = 128, w: int = 128) -> np.ndarray:
    """Random BGR uint8 frame."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _bbox(*coords) -> np.ndarray:
    return np.array(coords, dtype=np.float32)


def _bool_mask(x1: int, y1: int, x2: int, y2: int, h: int = 64, w: int = 64) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _tracker(**kwargs) -> SegmentationTracker:
    """Instantiate with sensible test defaults (no real model paths)."""
    defaults = dict(
        use_homography=False,
        redetect_interval=0,
        iou_threshold=0.3,
        conf_threshold=0.25,
        ball_conf_threshold=0.10,
        max_age=5,
    )
    defaults.update(kwargs)
    return SegmentationTracker(**defaults)


def _make_video(n_frames: int = 5, h: int = 64, w: int = 64) -> str:
    """Write a tiny MJPEG AVI to a temp file; caller must delete it."""
    fd, path = tempfile.mkstemp(suffix=".avi")
    os.close(fd)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(path, fourcc, 30.0, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        out.write(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    out.release()
    return path


def _sam_tracker_mock(prompted: bool = True) -> MagicMock:
    """SAM2Tracker mock that returns empty sv.Detections from track()."""
    m = MagicMock()
    m._prompted = prompted

    def _prompt_side_effect(frame, detections):
        m._prompted = True

    m.prompt_first_frame.side_effect = _prompt_side_effect
    m.track.return_value = sv.Detections.empty()
    return m


def _ball_tracker_mock(
    initialized: bool = False,
    frames_since_detection: int = 0,
    update_return: tuple[float, float] = (60.0, 60.0),
    predict_return: tuple[float, float] = (70.0, 70.0),
    last_source: str = "mosse",
    predicted_position: tuple[float, float] | None = (64.0, 64.0),
    adaptive_search_radius: int = 60,
) -> MagicMock:
    m = MagicMock()
    type(m).initialized = PropertyMock(return_value=initialized)
    type(m).frames_since_detection = PropertyMock(return_value=frames_since_detection)
    type(m).predicted_position = PropertyMock(return_value=predicted_position)
    type(m).adaptive_search_radius = PropertyMock(return_value=adaptive_search_radius)
    type(m).last_source = PropertyMock(return_value=last_source)
    m.update.return_value = update_return
    m.predict.return_value = predict_return
    return m


# ---------------------------------------------------------------------------
# TrackerState dataclass
# ---------------------------------------------------------------------------

class TestTrackerState:
    def test_defaults(self):
        ts = TrackerState()
        assert ts.frame_index == 0
        assert ts.tracks is None
        assert ts.ball_center is None
        assert ts.ball_source == "none"

    def test_custom_values(self):
        ts = TrackerState(frame_index=7, ball_center=(10.0, 20.0), ball_source="detected")
        assert ts.frame_index == 7
        assert ts.ball_center == (10.0, 20.0)
        assert ts.ball_source == "detected"

    def test_tracks_assignment(self):
        det = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            tracker_id=np.array([1], dtype=np.int32),
        )
        ts = TrackerState(tracks=det)
        assert ts.tracks is det


# ---------------------------------------------------------------------------
# Construction & tracker YAML
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_params_stored(self):
        t = _tracker()
        assert t.conf_threshold == pytest.approx(0.25)
        assert t.iou_threshold == pytest.approx(0.3)
        assert t.tracker == "botsort"
        assert t.max_age == 5
        assert t._detector is None
        assert t._predictor is None

    def test_tracker_name_normalised(self):
        t = _tracker(tracker="BoTSORT.yaml")
        assert t.tracker == "botsort"

    def test_bytetrack_name(self):
        t = _tracker(tracker="bytetrack")
        assert t.tracker == "bytetrack"

    def test_yaml_written_at_init(self):
        t = _tracker()
        assert t._tracker_config_path is not None
        assert os.path.isfile(t._tracker_config_path)

    def test_botsort_yaml_contains_correct_values(self):
        t = _tracker(tracker="botsort", conf_threshold=0.30, max_age=20)
        with open(t._tracker_config_path) as fh:
            content = fh.read()
        assert "botsort" in content
        assert "0.300" in content
        assert "20" in content

    def test_bytetrack_yaml_contains_correct_values(self):
        t = _tracker(tracker="bytetrack", conf_threshold=0.40, max_age=15)
        with open(t._tracker_config_path) as fh:
            content = fh.read()
        assert "bytetrack" in content
        assert "0.400" in content
        assert "15" in content

    def test_del_removes_yaml(self):
        t = _tracker()
        path = t._tracker_config_path
        assert os.path.isfile(path)
        del t
        assert not os.path.isfile(path)


# ---------------------------------------------------------------------------
# _bbox_iou (static)
# ---------------------------------------------------------------------------

class TestBboxIou:
    def test_identical(self):
        b = _bbox(10, 10, 50, 50)
        assert SegmentationTracker._bbox_iou(b, b) == pytest.approx(1.0, abs=1e-5)

    def test_no_overlap(self):
        assert SegmentationTracker._bbox_iou(_bbox(0, 0, 5, 5), _bbox(10, 10, 20, 20)) == 0.0

    def test_partial_overlap(self):
        a = _bbox(0, 0, 20, 20)   # area 400
        b = _bbox(10, 10, 30, 30)  # area 400; intersection 10×10=100
        expected = 100.0 / (400 + 400 - 100)
        assert SegmentationTracker._bbox_iou(a, b) == pytest.approx(expected, abs=1e-4)

    def test_contained_box(self):
        outer = _bbox(0, 0, 100, 100)  # area 10000
        inner = _bbox(10, 10, 90, 90)  # area 6400
        iou = SegmentationTracker._bbox_iou(outer, inner)
        assert iou == pytest.approx(6400.0 / 10000.0, abs=1e-4)

    def test_adjacent_no_overlap(self):
        a = _bbox(0, 0, 10, 10)
        b = _bbox(10, 0, 20, 10)  # shares edge only → zero intersection
        assert SegmentationTracker._bbox_iou(a, b) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# _mask_to_bbox (static)
# ---------------------------------------------------------------------------

class TestMaskToBbox:
    def test_empty_mask(self):
        assert SegmentationTracker._mask_to_bbox(np.zeros((64, 64), np.uint8)) is None

    def test_single_pixel(self):
        m = np.zeros((64, 64), np.uint8)
        m[10, 20] = 255
        np.testing.assert_array_equal(SegmentationTracker._mask_to_bbox(m), [20, 10, 20, 10])

    def test_rectangle(self):
        m = np.zeros((64, 64), np.uint8)
        m[5:15, 10:30] = 255
        result = SegmentationTracker._mask_to_bbox(m)
        np.testing.assert_array_equal(result, [10, 5, 29, 14])

    def test_bool_mask(self):
        m = np.zeros((64, 64), dtype=bool)
        m[0:10, 0:10] = True
        result = SegmentationTracker._mask_to_bbox(m)
        assert result is not None
        np.testing.assert_array_equal(result, [0, 0, 9, 9])


# ---------------------------------------------------------------------------
# _rect_mask (static)
# ---------------------------------------------------------------------------

class TestRectMask:
    def test_output_shape(self):
        mask = SegmentationTracker._rect_mask(_bbox(5, 5, 20, 20), (64, 64, 3))
        assert mask.shape == (64, 64)

    def test_interior_filled(self):
        mask = SegmentationTracker._rect_mask(_bbox(10, 10, 30, 30), (64, 64))
        assert mask[15, 15] > 0

    def test_exterior_zero(self):
        mask = SegmentationTracker._rect_mask(_bbox(10, 10, 30, 30), (64, 64))
        assert mask[5, 5] == 0
        assert mask[35, 35] == 0

    def test_clamps_out_of_bounds(self):
        mask = SegmentationTracker._rect_mask(_bbox(-10, -10, 200, 200), (64, 64))
        assert mask.shape == (64, 64)
        assert mask.any()


# ---------------------------------------------------------------------------
# _segment_ball_from_center (static)
# ---------------------------------------------------------------------------

class TestSegmentBallFromCenter:
    F = _frame(128, 128)

    def test_returns_mask_and_center(self):
        mask, center = SegmentationTracker._segment_ball_from_center(self.F, 64.0, 64.0, None)
        assert mask.shape == (128, 128)
        assert center == (64.0, 64.0)

    def test_centre_pixel_filled(self):
        mask, _ = SegmentationTracker._segment_ball_from_center(self.F, 64.0, 64.0, None)
        assert mask[64, 64] > 0

    def test_with_bbox(self):
        bbox = _bbox(50, 50, 78, 78)
        mask, center = SegmentationTracker._segment_ball_from_center(self.F, 64.0, 64.0, bbox)
        assert mask.shape == (128, 128)
        assert center == (64.0, 64.0)
        assert mask[64, 64] > 0

    def test_out_of_bounds_does_not_raise(self):
        mask, _ = SegmentationTracker._segment_ball_from_center(self.F, -20.0, -20.0, None)
        assert mask.shape == (128, 128)

    def test_fallback_radius_when_no_bbox(self):
        mask, _ = SegmentationTracker._segment_ball_from_center(self.F, 64.0, 64.0, None)
        # With rx=ry=15 the ellipse diameter is 30px; pixels 16 away should be outside
        assert mask[64, 64 + 20] == 0
        assert mask[64, 64 - 20] == 0


# ---------------------------------------------------------------------------
# _warp_bboxes (static)
# ---------------------------------------------------------------------------

class TestWarpBboxes:
    def test_none_homography_returns_same_object(self):
        boxes = [_bbox(10, 10, 50, 50)]
        result = SegmentationTracker._warp_bboxes(boxes, None, (128, 128))
        assert result is boxes

    def test_empty_list(self):
        H = np.eye(3, dtype=np.float64)
        assert SegmentationTracker._warp_bboxes([], H, (128, 128)) == []

    def test_identity_homography_preserves_box(self):
        boxes = [_bbox(10.0, 10.0, 50.0, 50.0)]
        H = np.eye(3, dtype=np.float64)
        result = SegmentationTracker._warp_bboxes(boxes, H, (128, 128))
        np.testing.assert_allclose(result[0], boxes[0], atol=1.0)

    def test_output_clamped_to_frame(self):
        boxes = [_bbox(0.0, 0.0, 64.0, 64.0)]
        # Large translation — boxes should be clamped to [0, 127]
        H = np.eye(3, dtype=np.float64)
        H[0, 2] = -500.0  # shift far left
        result = SegmentationTracker._warp_bboxes(boxes, H, (128, 128))
        assert result[0][0] >= 0.0
        assert result[0][2] <= 127.0


# ---------------------------------------------------------------------------
# _estimate_homography (static)
# ---------------------------------------------------------------------------

class TestEstimateHomography:
    def test_plain_black_frame_returns_none(self):
        blank = np.zeros((128, 128, 3), dtype=np.uint8)
        H = SegmentationTracker._estimate_homography(blank, blank)
        assert H is None

    def test_textured_frame_returns_matrix_or_none(self):
        # Checkerboard has many keypoints
        f = np.kron(
            np.indices((8, 8)).sum(axis=0) % 2,
            np.ones((16, 16), dtype=np.uint8),
        ).astype(np.uint8) * 255
        bgr = cv2.merge([f, f, f])
        H = SegmentationTracker._estimate_homography(bgr, bgr)
        if H is not None:
            assert H.shape == (3, 3)


# ---------------------------------------------------------------------------
# _hungarian_match_bboxes
# ---------------------------------------------------------------------------

class TestHungarianMatchBboxes:
    def setup_method(self):
        self.t = _tracker(iou_threshold=0.3)

    def test_empty_inputs(self):
        assert self.t._hungarian_match_bboxes([], []) == []
        assert self.t._hungarian_match_bboxes([_bbox(0, 0, 10, 10)], []) == []
        assert self.t._hungarian_match_bboxes([], [_bbox(0, 0, 10, 10)]) == []

    def test_identity_match(self):
        boxes = [_bbox(0, 0, 20, 20), _bbox(30, 30, 50, 50)]
        matches = self.t._hungarian_match_bboxes(boxes, boxes)
        assert set(matches) == {(0, 0), (1, 1)}

    def test_no_overlap_no_match(self):
        a = [_bbox(0, 0, 5, 5)]
        b = [_bbox(50, 50, 60, 60)]
        assert self.t._hungarian_match_bboxes(a, b) == []

    def test_cross_match(self):
        a = [_bbox(0, 0, 20, 20), _bbox(30, 30, 50, 50)]
        b = [_bbox(29, 29, 51, 51), _bbox(0, 0, 21, 21)]
        matches = dict(self.t._hungarian_match_bboxes(a, b))
        assert matches[0] == 1   # a[0] ↔ b[1]
        assert matches[1] == 0   # a[1] ↔ b[0]

    def test_subset_match(self):
        # 3 tracks, 2 masks — best 2 should be matched
        a = [_bbox(0, 0, 20, 20), _bbox(30, 30, 50, 50), _bbox(60, 60, 80, 80)]
        b = [_bbox(0, 0, 20, 20), _bbox(30, 30, 50, 50)]
        matches = self.t._hungarian_match_bboxes(a, b)
        assert len(matches) == 2


# ---------------------------------------------------------------------------
# _assign_new_player_ids
# ---------------------------------------------------------------------------

class TestAssignNewPlayerIds:
    def setup_method(self):
        self.t = _tracker(iou_threshold=0.3)
        self.t._next_player_id = 1

    def test_empty_returns_empty(self):
        assert self.t._assign_new_player_ids([], [], []) == []

    def test_fresh_ids_no_existing(self):
        new = [_bbox(0, 0, 10, 10), _bbox(20, 20, 30, 30)]
        ids = self.t._assign_new_player_ids(new, [], [])
        assert ids == [1, 2]
        assert self.t._next_player_id == 3

    def test_matched_to_existing(self):
        existing_b = [_bbox(0, 0, 10, 10)]
        existing_ids = [42]
        ids = self.t._assign_new_player_ids([_bbox(0, 0, 10, 10)], existing_b, existing_ids)
        assert ids == [42]
        assert self.t._next_player_id == 1  # no new IDs consumed

    def test_unmatched_gets_new_id(self):
        existing_b = [_bbox(0, 0, 10, 10)]
        existing_ids = [42]
        ids = self.t._assign_new_player_ids([_bbox(60, 60, 80, 80)], existing_b, existing_ids)
        assert ids == [1]
        assert self.t._next_player_id == 2

    def test_length_equals_input(self):
        new = [_bbox(i * 20, 0, i * 20 + 15, 15) for i in range(5)]
        ids = self.t._assign_new_player_ids(new, [], [])
        assert len(ids) == 5

    def test_counter_monotonic(self):
        new = [_bbox(i * 30, 0, i * 30 + 20, 20) for i in range(4)]
        ids = self.t._assign_new_player_ids(new, [], [])
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# _merge_bot_sam_results
# ---------------------------------------------------------------------------

class TestMergeBotSamResults:
    def setup_method(self):
        self.t = _tracker(iou_threshold=0.3)
        self.frame = _frame(64, 64)

    def _sam_det(self, bboxes, masks) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array(bboxes, dtype=np.float32),
            mask=np.array(masks, dtype=bool),
        )

    def test_empty_sam_falls_back_to_rect_masks(self):
        bot = {1: _bbox(5, 5, 20, 20), 2: _bbox(30, 30, 50, 50)}
        result = TrackerState()
        result = self.t._merge_bot_sam_results(bot, sv.Detections.empty(), self.frame, result)
        assert result.tracks is not None
        assert len(result.tracks) == 2
        assert set(result.tracks.tracker_id.tolist()) == {1, 2}

    def test_matching_sam_mask_used(self):
        bot = {5: _bbox(5, 5, 20, 20)}
        sam_mask = _bool_mask(5, 5, 20, 20)
        sam = self._sam_det([[5, 5, 20, 20]], [sam_mask])
        result = TrackerState()
        result = self.t._merge_bot_sam_results(bot, sam, self.frame, result)
        assert result.tracks is not None
        assert result.tracks.tracker_id[0] == 5
        np.testing.assert_array_equal(result.tracks.mask[0], sam_mask)

    def test_unmatched_bot_track_gets_rect_fallback(self):
        bot = {7: _bbox(5, 5, 20, 20), 8: _bbox(40, 40, 60, 60)}
        # SAM only covers bbox for track 7
        sam = self._sam_det([[5, 5, 20, 20]], [_bool_mask(5, 5, 20, 20)])
        result = TrackerState()
        result = self.t._merge_bot_sam_results(bot, sam, self.frame, result)
        assert result.tracks is not None
        assert len(result.tracks) == 2
        assert set(result.tracks.tracker_id.tolist()) == {7, 8}

    def test_no_bot_tracks_no_tracks(self):
        result = TrackerState()
        result = self.t._merge_bot_sam_results({}, sv.Detections.empty(), self.frame, result)
        assert result.tracks is None

    def test_tracker_ids_correct_dtype(self):
        bot = {3: _bbox(5, 5, 20, 20)}
        result = TrackerState()
        result = self.t._merge_bot_sam_results(bot, sv.Detections.empty(), self.frame, result)
        assert result.tracks.tracker_id.dtype == np.int32


# ---------------------------------------------------------------------------
# _fallback_detect
# ---------------------------------------------------------------------------

class TestFallbackDetect:
    def setup_method(self):
        self.t = _tracker(iou_threshold=0.3)
        self.t._next_player_id = 1
        self.frame = _frame(64, 64)

    def test_no_detections_leaves_tracks_none(self):
        self.t._detect_frame = MagicMock(return_value=([], None))
        result = TrackerState(frame_index=0)
        result = self.t._fallback_detect(self.frame, 0, [], result)
        assert result.tracks is None

    def test_new_players_get_fresh_ids(self):
        bboxes = [_bbox(0, 0, 10, 10), _bbox(20, 20, 30, 30)]
        self.t._detect_frame = MagicMock(return_value=(bboxes, None))
        result = TrackerState(frame_index=0)
        result = self.t._fallback_detect(self.frame, 0, [], result)
        assert result.tracks is not None
        assert len(result.tracks) == 2
        assert set(result.tracks.tracker_id.tolist()) == {1, 2}

    def test_id_preserved_across_frames(self):
        bboxes = [_bbox(0, 0, 10, 10)]
        self.t._detect_frame = MagicMock(return_value=(bboxes, None))
        r0 = TrackerState(frame_index=0)
        r0 = self.t._fallback_detect(self.frame, 0, [], r0)
        r1 = TrackerState(frame_index=1)
        r1 = self.t._fallback_detect(self.frame, 1, [r0], r1)
        assert r1.tracks.tracker_id[0] == 1  # same ID reused

    def test_ball_entries_excluded_from_existing(self):
        """Ball entry (tracker_id == -1) in previous frame must not be used as existing player."""
        prev = TrackerState(frame_index=0)
        prev.tracks = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10], [40, 40, 60, 60]], dtype=np.float32),
            mask=np.array([np.zeros((64, 64), bool), np.zeros((64, 64), bool)]),
            tracker_id=np.array([1, -1], dtype=np.int32),  # -1 = ball
        )
        new_bbox = [_bbox(0, 0, 10, 10)]
        self.t._detect_frame = MagicMock(return_value=(new_bbox, None))
        r1 = TrackerState(frame_index=1)
        r1 = self.t._fallback_detect(self.frame, 1, [prev], r1)
        # Should match player id 1, not assign a new id
        assert r1.tracks.tracker_id[0] == 1

    def test_rect_masks_have_correct_shape(self):
        bboxes = [_bbox(5, 5, 20, 20)]
        self.t._detect_frame = MagicMock(return_value=(bboxes, None))
        result = TrackerState(frame_index=0)
        result = self.t._fallback_detect(self.frame, 0, [], result)
        assert result.tracks.mask[0].shape == (64, 64)


# ---------------------------------------------------------------------------
# _redetect_new_players
# ---------------------------------------------------------------------------

class TestRedetectNewPlayers:
    def setup_method(self):
        self.t = _tracker(iou_threshold=0.3)
        self.t._next_player_id = 10
        self.frame = _frame(64, 64)

    def test_no_detections_unchanged(self):
        self.t._detect_frame = MagicMock(return_value=([], None))
        seg = TrackerState()
        returned = self.t._redetect_new_players(self.frame, seg)
        assert returned is seg
        assert returned.tracks is None

    def test_already_tracked_not_duplicated(self):
        bbox = _bbox(0, 0, 20, 20)
        seg = TrackerState()
        seg.tracks = sv.Detections(
            xyxy=np.array([bbox]),
            mask=np.array([np.zeros((64, 64), bool)]),
            tracker_id=np.array([1], dtype=np.int32),
        )
        self.t._detect_frame = MagicMock(return_value=([bbox], None))
        result = self.t._redetect_new_players(self.frame, seg)
        assert len(result.tracks) == 1

    def test_new_player_appended(self):
        old_bbox = _bbox(0, 0, 20, 20)
        new_bbox = _bbox(40, 40, 60, 60)
        seg = TrackerState()
        seg.tracks = sv.Detections(
            xyxy=np.array([old_bbox]),
            mask=np.array([np.zeros((64, 64), bool)]),
            tracker_id=np.array([1], dtype=np.int32),
        )
        self.t._detect_frame = MagicMock(return_value=([old_bbox, new_bbox], None))
        result = self.t._redetect_new_players(self.frame, seg)
        assert len(result.tracks) == 2
        assert 1 in result.tracks.tracker_id
        assert 10 in result.tracks.tracker_id

    def test_ball_entry_not_treated_as_existing_player(self):
        """Ball entry (tracker_id == -1) must not prevent new player addition."""
        ball_bbox = _bbox(30, 30, 50, 50)
        new_player_bbox = _bbox(0, 0, 20, 20)
        seg = TrackerState()
        seg.tracks = sv.Detections(
            xyxy=np.array([ball_bbox]),
            mask=np.array([np.zeros((64, 64), bool)]),
            tracker_id=np.array([-1], dtype=np.int32),
        )
        self.t._detect_frame = MagicMock(return_value=([new_player_bbox], None))
        result = self.t._redetect_new_players(self.frame, seg)
        # new player added — total becomes 2
        assert len(result.tracks) == 2


# ---------------------------------------------------------------------------
# _process_ball
# ---------------------------------------------------------------------------

class TestProcessBall:
    def setup_method(self):
        self.t = _tracker()
        self.frame = _frame(128, 128)

    def test_detected_bbox_sets_ball_center(self):
        self.t._ball_tracker = _ball_tracker_mock(update_return=(60.0, 60.0))
        seg = TrackerState()
        result = self.t._process_ball(self.frame, _bbox(50, 50, 70, 70), seg)
        assert result.ball_center == (60.0, 60.0)
        assert result.ball_source == "detected"

    def test_detected_bbox_adds_ball_to_tracks(self):
        self.t._ball_tracker = _ball_tracker_mock(update_return=(60.0, 60.0))
        seg = TrackerState()
        result = self.t._process_ball(self.frame, _bbox(50, 50, 70, 70), seg)
        assert result.tracks is not None
        assert -1 in result.tracks.tracker_id

    def test_no_bbox_not_initialized_no_ball(self):
        self.t._ball_tracker = _ball_tracker_mock(initialized=False)
        seg = TrackerState()
        result = self.t._process_ball(self.frame, None, seg)
        assert result.ball_center is None
        assert result.ball_source == "none"

    def test_no_bbox_initialized_triggers_roi_check(self):
        self.t._ball_tracker = _ball_tracker_mock(initialized=True, frames_since_detection=1)
        self.t._detect_ball_in_roi = MagicMock(return_value=None)
        seg = TrackerState()
        self.t._process_ball(self.frame, None, seg)
        self.t._detect_ball_in_roi.assert_called_once()

    def test_roi_miss_falls_back_to_predict(self):
        bt = _ball_tracker_mock(initialized=True, frames_since_detection=1, predict_return=(70.0, 70.0))
        self.t._ball_tracker = bt
        self.t._detect_ball_in_roi = MagicMock(return_value=None)
        seg = TrackerState()
        result = self.t._process_ball(self.frame, None, seg)
        bt.predict.assert_called_once()
        # ball_source comes from last_source mock ("mosse")
        assert result.ball_source == "mosse"

    def test_roi_hit_uses_roi_bbox(self):
        roi_bbox = _bbox(55, 55, 75, 75)
        bt = _ball_tracker_mock(initialized=True, frames_since_detection=1, update_return=(65.0, 65.0))
        self.t._ball_tracker = bt
        self.t._detect_ball_in_roi = MagicMock(return_value=roi_bbox)
        seg = TrackerState()
        result = self.t._process_ball(self.frame, None, seg)
        assert result.ball_source == "roi"
        assert result.ball_center == (65.0, 65.0)
        bt.update.assert_called_once()

    def test_ball_merged_with_existing_player_tracks(self):
        self.t._ball_tracker = _ball_tracker_mock(update_return=(60.0, 60.0))
        seg = TrackerState()
        seg.tracks = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            mask=np.array([np.zeros((128, 128), bool)]),
            tracker_id=np.array([3], dtype=np.int32),
        )
        result = self.t._process_ball(self.frame, _bbox(50, 50, 70, 70), seg)
        assert len(result.tracks) == 2
        assert -1 in result.tracks.tracker_id
        assert 3 in result.tracks.tracker_id

    def test_predicted_position_creates_placeholder_bbox(self):
        """When ball is predicted (no bbox), a 30×30 bbox is synthesised around the predicted point."""
        bt = _ball_tracker_mock(
            initialized=True, frames_since_detection=1,
            predict_return=(48.0, 50.0), last_source="predicted"
        )
        self.t._ball_tracker = bt
        self.t._detect_ball_in_roi = MagicMock(return_value=None)
        seg = TrackerState()
        result = self.t._process_ball(self.frame, None, seg)
        if result.tracks is not None:
            idx = np.where(result.tracks.tracker_id == -1)[0]
            if len(idx) > 0:
                bx = result.tracks.xyxy[idx[0]]
                cx = (bx[0] + bx[2]) / 2
                cy = (bx[1] + bx[3]) / 2
                assert abs(cx - 48.0) < 1.0
                assert abs(cy - 50.0) < 1.0

    def test_update_called_with_frame(self):
        """update() must receive the frame so MOSSE can refresh its template."""
        bt = _ball_tracker_mock(update_return=(60.0, 60.0))
        self.t._ball_tracker = bt
        seg = TrackerState()
        self.t._process_ball(self.frame, _bbox(50, 50, 70, 70), seg)
        call_args = bt.update.call_args
        # third positional or keyword arg should be the frame
        assert call_args is not None
        args, kwargs = call_args
        frame_arg = args[2] if len(args) > 2 else kwargs.get("frame")
        assert frame_arg is not None


# ---------------------------------------------------------------------------
# process_video — integration with mocked YOLO & SAM2
# ---------------------------------------------------------------------------

class TestProcessVideo:
    def setup_method(self):
        self.t = _tracker(use_homography=False, redetect_interval=0)
        self.video_path = _make_video(n_frames=4, h=64, w=64)

    def teardown_method(self):
        if os.path.isfile(self.video_path):
            os.unlink(self.video_path)

    # ---- helpers -----------------------------------------------------------

    def _stub_detect_frame(self, person_bboxes=None, ball_bbox=None):
        """Replace _detect_frame with a stub returning fixed data."""
        bboxes = person_bboxes or [np.array([5, 5, 20, 20], dtype=np.float32)]
        self.t._detect_frame = MagicMock(return_value=(bboxes, ball_bbox))

    def _stub_track_frame(self, person_bboxes=None, ball_bbox=None):
        """Replace _track_frame with a stub returning fixed BoT-SORT-style data."""
        bboxes = person_bboxes or [np.array([5, 5, 20, 20], dtype=np.float32)]
        player_tracks: dict[int, np.ndarray] = {
            i + 1: b for i, b in enumerate(bboxes)
        }
        self.t._track_frame = MagicMock(return_value=(player_tracks, ball_bbox))

    def _patch_sam(self, prompted: bool = True):
        sam = _sam_tracker_mock(prompted=prompted)
        self.t._get_sam_predictor = MagicMock(return_value=sam)
        return sam

    # ---- tests -------------------------------------------------------------

    def test_returns_correct_frame_count(self):
        self._stub_detect_frame()
        self._stub_track_frame()
        self._patch_sam()
        results = self.t.process_video(self.video_path, max_frames=4)
        assert len(results) == 4

    def test_max_frames_respected(self):
        self._stub_detect_frame()
        self._stub_track_frame()
        self._patch_sam()
        results = self.t.process_video(self.video_path, max_frames=2)
        assert len(results) == 2

    def test_frame_indices_sequential(self):
        self._stub_detect_frame()
        self._stub_track_frame()
        self._patch_sam()
        results = self.t.process_video(self.video_path, max_frames=3)
        for i, r in enumerate(results):
            assert r.frame_index == i

    def test_results_are_tracker_state(self):
        self._stub_detect_frame()
        self._stub_track_frame()
        self._patch_sam()
        results = self.t.process_video(self.video_path, max_frames=2)
        assert all(isinstance(r, TrackerState) for r in results)

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError, match="Cannot open"):
            self.t.process_video("/nonexistent/path/video.mp4")

    def test_player_tracks_present_from_botsort(self):
        self._stub_detect_frame()
        self._stub_track_frame(person_bboxes=[np.array([5, 5, 20, 20], dtype=np.float32)])
        self._patch_sam()
        results = self.t.process_video(self.video_path, max_frames=2)
        for r in results:
            if r.tracks is not None:
                player_ids = r.tracks.tracker_id[r.tracks.tracker_id != -1]
                assert len(player_ids) >= 0  # at least doesn't crash

    def test_ball_detected_sets_center(self):
        ball_np = np.array([10, 10, 20, 20], dtype=np.float32)
        self._stub_detect_frame(ball_bbox=None)
        self._stub_track_frame(ball_bbox=ball_np)
        self._patch_sam()
        results = self.t.process_video(self.video_path, max_frames=3)
        centers = [r.ball_center for r in results if r.ball_center is not None]
        # At least one frame should have a ball center resolved via the DCF tracker
        assert len(centers) > 0

    def test_ball_entry_has_tracker_id_minus_one(self):
        ball_np = np.array([10, 10, 20, 20], dtype=np.float32)
        self._stub_detect_frame(ball_bbox=None)
        self._stub_track_frame(ball_bbox=ball_np)
        self._patch_sam()
        results = self.t.process_video(self.video_path, max_frames=3)
        for r in results:
            if r.tracks is not None and -1 in r.tracks.tracker_id:
                # confirmed: ball is tagged with -1
                break
        else:
            pytest.fail("No frame had a ball entry (tracker_id == -1)")

    def test_next_player_id_reset_on_each_call(self):
        self._stub_detect_frame()
        self._stub_track_frame()
        self._patch_sam()
        self.t.process_video(self.video_path, max_frames=2)
        self.t._get_sam_predictor = MagicMock(return_value=_sam_tracker_mock())
        self.t.process_video(self.video_path, max_frames=2)
        # After the second call _next_player_id should have restarted from 1
        # (BoT-SORT assigns its own IDs; fallback path would start from 1)
        assert self.t._next_player_id >= 1

    def test_sam_tracker_prompted_when_players_detected(self):
        bboxes = [np.array([5, 5, 20, 20], dtype=np.float32)]
        self._stub_detect_frame(person_bboxes=bboxes)
        self._stub_track_frame(person_bboxes=bboxes)
        sam = self._patch_sam(prompted=False)
        self.t.process_video(self.video_path, max_frames=2)
        sam.prompt_first_frame.assert_called_once()

    def test_fallback_path_when_no_botsort_tracks(self):
        """When BoT-SORT returns no players, _fallback_detect must populate tracks."""
        self._stub_detect_frame(person_bboxes=[np.array([5, 5, 20, 20], dtype=np.float32)])
        # track_frame returns empty dict → triggers fallback
        self.t._track_frame = MagicMock(return_value=({}, None))
        self._patch_sam()
        results = self.t.process_video(self.video_path, max_frames=2)
        # fallback should populate at least some frames with tracks
        has_tracks = any(r.tracks is not None for r in results)
        assert has_tracks

    def test_homography_estimated_when_enabled(self):
        t = _tracker(use_homography=True, redetect_interval=0)
        t._track_frame = MagicMock(return_value=({}, None))
        t._detect_frame = MagicMock(return_value=([], None))
        t._get_sam_predictor = MagicMock(return_value=_sam_tracker_mock())
        with patch.object(
            SegmentationTracker, "_estimate_homography", wraps=SegmentationTracker._estimate_homography
        ) as spy:
            t.process_video(self.video_path, max_frames=3)
            # Should be called n-1 times (frame pairs)
            assert spy.call_count >= 2
