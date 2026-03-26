"""
tests/test_association.py
--------------------------
Unit tests for segmentation_tracking/association.py.

No GPU, no pretrained weights, and no external downloads are required.
YOLO-style tensors are mocked with plain torch tensors; sv.Detections is
built from NumPy arrays.

Known bugs documented in TestKnownBugs:
  1. Ball-track building accesses ``seg_result.xyxy[-1]`` and
     ``seg_result.mask[-1]`` instead of ``seg_result.tracks.xyxy[-1]``/
     ``seg_result.tracks.mask[-1]``.  TrackerState has no top-level ``.xyxy``
     or ``.mask`` attributes, so passing a real TrackerState with
     ``ball_center`` set raises AttributeError.  Tests that exercise the ball
     path use ``types.SimpleNamespace`` (with top-level ``.xyxy`` / ``.mask``)
     to work around this.
  2. ``associate_poses_with_tracks`` calls ``seg_result.tracks.tracker_id``
     without guarding against ``tracks=None``, raising AttributeError.

Run with:
    pytest tests/test_association.py -v
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest
import torch
import supervision as sv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from segmentation_tracking.association import (  # noqa: E402
    _bbox_iou,
    _center_distance,
    PlayerTrack,
    BallTrack,
    associate_poses_with_tracks,
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
)
from segmentation_tracking.segmentation_model import TrackerState  # noqa: E402


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _b(*coords) -> np.ndarray:
    return np.array(coords, dtype=np.float32)


def _mask(x1: int, y1: int, x2: int, y2: int, h: int = 64, w: int = 64) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _kps(seed: int = 7) -> np.ndarray:
    """Random (17, 2) float32 keypoints inside a 64×64 frame."""
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 63, (17, 2)).astype(np.float32)


def _scores(val: float = 0.9) -> np.ndarray:
    return np.full((17,), val, dtype=np.float32)


def _detections(bboxes, tracker_ids, masks=None, H: int = 64, W: int = 64) -> sv.Detections:
    xyxy = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
    tids = np.array(tracker_ids, dtype=int)
    if masks is None:
        masks = np.zeros((len(bboxes), H, W), dtype=bool)
    return sv.Detections(xyxy=xyxy, mask=np.array(masks), tracker_id=tids)


def _seg(bboxes=None, tracker_ids=None, masks=None,
         ball_center=None, ball_source="none") -> TrackerState:
    """Build a TrackerState. ball_center should be None to avoid the known bug."""
    tracks = _detections(bboxes, tracker_ids, masks) if bboxes else None
    return TrackerState(
        frame_index=0,
        tracks=tracks,
        ball_center=ball_center,
        ball_source=ball_source,
    )


def _seg_with_ball(bboxes, tracker_ids, masks, ball_center,
                   ball_source: str = "detected"):
    """
    SimpleNamespace workaround for the bug where associate_poses_with_tracks
    accesses ``seg_result.xyxy[-1]`` and ``seg_result.mask[-1]`` on the
    TrackerState object directly instead of going through ``.tracks``.
    """
    tracks = _detections(bboxes, tracker_ids, masks)
    return types.SimpleNamespace(
        frame_index=0,
        tracks=tracks,
        ball_center=ball_center,
        ball_source=ball_source,
        field_mask=None,
        xyxy=tracks.xyxy,   # top-level so seg_result.xyxy[-1] works
        mask=tracks.mask,   # top-level so seg_result.mask[-1] works
    )


def _new_pose(bboxes, keypoints, scores=None):
    """New pose_estimation.PoseResult format (presence of num_persons triggers it)."""
    n = len(bboxes)
    if scores is None:
        scores = np.ones((n, 17), dtype=np.float32)
    if n == 0:
        return types.SimpleNamespace(
            num_persons=0,
            bboxes=np.zeros((0, 4), dtype=np.float32),
            keypoints=np.zeros((0, 17, 2), dtype=np.float32),
            scores=np.zeros((0, 17), dtype=np.float32),
        )
    return types.SimpleNamespace(
        num_persons=n,
        bboxes=np.array(bboxes, dtype=np.float32).reshape(-1, 4),
        keypoints=np.array(keypoints, dtype=np.float32).reshape(n, 17, 2),
        scores=np.array(scores, dtype=np.float32).reshape(n, 17),
    )


class _YoloPose:
    """Minimal mock of a single ultralytics YOLO pose result.

    Deliberately has no ``num_persons`` attribute so the YOLO branch is taken.
    """

    def __init__(self, bboxes, keypoints, scores=None,
                 no_boxes: bool = False, no_conf: bool = False):
        n = len(bboxes)
        kps_t = torch.tensor(
            np.array(keypoints, dtype=np.float32).reshape(n, 17, 2)
        )
        bboxes_t = torch.tensor(
            np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        )
        if scores is None:
            conf_t = torch.ones(n, 17, dtype=torch.float32)
        else:
            conf_t = torch.tensor(
                np.array(scores, dtype=np.float32).reshape(n, 17)
            )

        class _KPS:
            def __len__(self_):       return n
            @property
            def xy(self_):            return kps_t
            @property
            def conf(self_):          return None if no_conf else conf_t

        class _Boxes:
            def __len__(self_):       return n
            @property
            def xyxy(self_):          return bboxes_t

        self.keypoints = _KPS()
        self.boxes     = None if no_boxes else _Boxes()


# ---------------------------------------------------------------------------
# TestBboxIou
# ---------------------------------------------------------------------------

class TestBboxIou:
    def test_identical_boxes_give_one(self):
        a = _b(0, 0, 10, 10)
        assert _bbox_iou(a, a) == pytest.approx(1.0, abs=1e-5)

    def test_non_overlapping_gives_zero(self):
        a = _b(0, 0, 10, 10)
        b = _b(20, 20, 30, 30)
        assert _bbox_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap_known_value(self):
        # Two 10×10 boxes sharing a 5×10 strip:
        #   intersection = 50, union = 200 - 50 = 150
        a = _b(0, 0, 10, 10)
        b = _b(5, 0, 15, 10)
        assert _bbox_iou(a, b) == pytest.approx(50.0 / 150.0, abs=1e-4)

    def test_inner_box_inside_outer(self):
        # 4×4 inner inside 10×10 outer: intersection = 16, union = 100
        outer = _b(0, 0, 10, 10)
        inner = _b(3, 3, 7, 7)
        assert _bbox_iou(outer, inner) == pytest.approx(16.0 / 100.0, abs=1e-4)

    def test_touching_edge_gives_zero(self):
        a = _b(0, 0, 10, 10)
        b = _b(10, 0, 20, 10)
        assert _bbox_iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_symmetry(self):
        a = _b(0, 0, 10, 20)
        b = _b(5, 10, 25, 30)
        assert _bbox_iou(a, b) == pytest.approx(_bbox_iou(b, a))

    def test_zero_area_box_gives_zero(self):
        a = _b(5, 5, 5, 5)      # degenerate: width = height = 0
        b = _b(0, 0, 10, 10)
        assert _bbox_iou(a, b) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestCenterDistance
# ---------------------------------------------------------------------------

class TestCenterDistance:
    def test_same_box_gives_zero(self):
        a = _b(0, 0, 10, 10)
        assert _center_distance(a, a) == pytest.approx(0.0)

    def test_horizontal_separation(self):
        # center(0,0,10,10)=(5,5), center(20,0,30,10)=(25,5) → dist=20
        a = _b(0, 0, 10, 10)
        b = _b(20, 0, 30, 10)
        assert _center_distance(a, b) == pytest.approx(20.0)

    def test_345_right_triangle(self):
        # center(0,0,0,0)=(0,0), center(6,8,6,8)=(6,8) → dist=10
        a = _b(0, 0, 0, 0)
        b = _b(6, 8, 6, 8)
        assert _center_distance(a, b) == pytest.approx(10.0)

    def test_symmetry(self):
        a = _b(0, 0, 10, 10)
        b = _b(5, 5, 40, 40)
        assert _center_distance(a, b) == pytest.approx(_center_distance(b, a))


# ---------------------------------------------------------------------------
# TestPlayerTrack
# ---------------------------------------------------------------------------

class TestPlayerTrack:
    def test_required_fields_stored(self):
        m = _mask(0, 0, 4, 4)
        b = _b(0, 0, 4, 4)
        pt = PlayerTrack(id=3, mask=m, bbox=b)
        assert pt.id == 3
        np.testing.assert_array_equal(pt.mask, m)
        np.testing.assert_array_equal(pt.bbox, b)

    def test_optional_fields_default_to_none(self):
        pt = PlayerTrack(id=1, mask=_mask(0, 0, 4, 4), bbox=_b(0, 0, 4, 4))
        assert pt.keypoints is None
        assert pt.keypoint_scores is None
        assert pt.team_label is None

    def test_to_dict_with_keypoints(self):
        kps = _kps()
        sc  = _scores()
        pt  = PlayerTrack(id=5, mask=_mask(0, 0, 4, 4), bbox=_b(1, 2, 3, 4),
                          keypoints=kps, keypoint_scores=sc, team_label=0)
        d = pt.to_dict()
        assert d["player_id"] == 5
        assert d["team_label"] == 0
        assert len(d["keypoints"]) == 17
        assert len(d["keypoint_scores"]) == 17
        assert d["bbox"] == pytest.approx([1, 2, 3, 4])

    def test_to_dict_without_keypoints_gives_none(self):
        pt = PlayerTrack(id=2, mask=_mask(0, 0, 4, 4), bbox=_b(0, 0, 4, 4))
        d = pt.to_dict()
        assert d["keypoints"] is None
        assert d["keypoint_scores"] is None
        assert d["team_label"] is None

    def test_team_label_can_be_set_externally(self):
        pt = PlayerTrack(id=1, mask=_mask(0, 0, 4, 4), bbox=_b(0, 0, 4, 4))
        pt.team_label = 2
        assert pt.team_label == 2


# ---------------------------------------------------------------------------
# TestBallTrack
# ---------------------------------------------------------------------------

class TestBallTrack:
    def test_minimal_construction(self):
        bt = BallTrack(center=(10.0, 20.0))
        assert bt.center == (10.0, 20.0)
        assert bt.bbox is None
        assert bt.mask is None
        assert bt.source == "none"

    def test_to_dict_full(self):
        bbox = _b(8, 18, 12, 22)
        bt   = BallTrack(center=(10.0, 20.0), bbox=bbox, source="detected")
        d    = bt.to_dict()
        assert d["ball_center"] == [10.0, 20.0]
        assert d["bbox"] == pytest.approx([8, 18, 12, 22])
        assert d["source"] == "detected"

    def test_to_dict_no_bbox_gives_none(self):
        bt = BallTrack(center=(5.0, 5.0))
        assert bt.to_dict()["bbox"] is None

    @pytest.mark.parametrize("src", ["detected", "roi", "mosse", "predicted", "none"])
    def test_all_source_values(self, src):
        bt = BallTrack(center=(0.0, 0.0), source=src)
        assert bt.to_dict()["source"] == src


# ---------------------------------------------------------------------------
# TestCocoConstants
# ---------------------------------------------------------------------------

class TestCocoConstants:
    def test_keypoint_names_count(self):
        assert len(COCO_KEYPOINT_NAMES) == 17

    def test_skeleton_indices_in_range(self):
        for a, b in COCO_SKELETON:
            assert 0 <= a < 17, f"skeleton index {a} out of range"
            assert 0 <= b < 17, f"skeleton index {b} out of range"


# ---------------------------------------------------------------------------
# TestKnownBugs  — document AttributeErrors in the current implementation
# ---------------------------------------------------------------------------

class TestBugFixes:
    """Verify previously known bugs are resolved."""

    def test_ball_only_entry_does_not_appear_in_player_tracks(self):
        """
        Ball entry (tracker_id == -1) must be excluded from player tracks
        and must populate the BallTrack instead.
        """
        seg = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[-1],
                   ball_center=(5.0, 5.0))
        pts, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert pts == [], "ball entry must not appear as a PlayerTrack"
        assert bt is not None
        assert bt.center == (5.0, 5.0)
        assert bt.bbox is not None
        np.testing.assert_array_almost_equal(bt.bbox, [0, 0, 10, 10])

    def test_none_tracks_returns_empty_results(self):
        """
        TrackerState with tracks=None must return empty lists, not raise.
        """
        seg = TrackerState(frame_index=0, tracks=None)
        pts, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert pts == []
        assert bt is None


# ---------------------------------------------------------------------------
# TestAssociateNoPose
# ---------------------------------------------------------------------------

class TestAssociateNoPose:
    def test_none_pose_all_tracks_have_no_keypoints(self):
        seg = _seg(bboxes=[[0, 0, 10, 10], [20, 20, 30, 30]], tracker_ids=[1, 2])
        pts, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert len(pts) == 2
        assert all(p.keypoints is None for p in pts)
        assert all(p.keypoint_scores is None for p in pts)
        assert bt is None

    def test_no_ball_center_gives_none_ball_track(self):
        seg = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1], ball_center=None)
        _, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert bt is None

    def test_pose_result_without_keypoints_attr_treated_as_no_pose(self):
        seg = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        pose = types.SimpleNamespace()          # no 'keypoints', no 'num_persons'
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert len(pts) == 1
        assert pts[0].keypoints is None

    def test_track_ids_preserved(self):
        seg = _seg(
            bboxes=[[0, 0, 10, 10], [20, 0, 30, 10], [50, 0, 60, 10]],
            tracker_ids=[7, 3, 11],
        )
        pts, _ = associate_poses_with_tracks(seg, None, (64, 64))
        assert [p.id for p in pts] == [7, 3, 11]

    def test_result_length_equals_track_count(self):
        for n in (1, 3, 5):
            bboxes      = [[i * 15, 0, i * 15 + 10, 10] for i in range(n)]
            tracker_ids = list(range(1, n + 1))
            seg  = _seg(bboxes=bboxes, tracker_ids=tracker_ids)
            pts, _ = associate_poses_with_tracks(seg, None, (64, 64))
            assert len(pts) == n

    def test_return_is_two_tuple(self):
        seg = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        result = associate_poses_with_tracks(seg, None, (64, 64))
        assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# TestBallTrackBuilding  (all use SimpleNamespace workaround for known bug)
# ---------------------------------------------------------------------------

class TestBallTrackBuilding:
    def test_ball_center_creates_ball_track(self):
        seg = _seg_with_ball(
            bboxes=[[0, 0, 10, 10], [50, 50, 60, 60]],
            tracker_ids=[1, -1],
            masks=[_mask(0, 0, 10, 10), _mask(50, 50, 60, 60)],
            ball_center=(55.0, 55.0),
            ball_source="detected",
        )
        _, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert bt is not None
        assert bt.center == (55.0, 55.0)

    def test_ball_source_forwarded(self):
        for src in ("detected", "roi", "mosse", "predicted"):
            seg = _seg_with_ball(
                bboxes=[[0, 0, 10, 10]],
                tracker_ids=[-1],
                masks=[_mask(0, 0, 10, 10)],
                ball_center=(5.0, 5.0),
                ball_source=src,
            )
            _, bt = associate_poses_with_tracks(seg, None, (64, 64))
            assert bt.source == src

    def test_ball_track_is_ball_track_instance(self):
        seg = _seg_with_ball(
            bboxes=[[0, 0, 10, 10]],
            tracker_ids=[-1],
            masks=[_mask(0, 0, 10, 10)],
            ball_center=(5.0, 5.0),
        )
        _, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert isinstance(bt, BallTrack)

    def test_no_ball_center_still_gives_none(self):
        seg = _seg_with_ball(
            bboxes=[[0, 0, 10, 10]],
            tracker_ids=[1],
            masks=[_mask(0, 0, 10, 10)],
            ball_center=None,
        )
        _, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert bt is None


# ---------------------------------------------------------------------------
# TestIoUAssociationNewFormat  — pose_estimation.PoseResult duck-type
# ---------------------------------------------------------------------------

class TestIoUAssociationNewFormat:
    def test_single_player_perfect_overlap_matched(self):
        bbox = [5, 5, 25, 25]
        seg  = _seg(bboxes=[bbox], tracker_ids=[1])
        pose = _new_pose(bboxes=[bbox], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert pts[0].keypoints is not None

    def test_single_player_no_overlap_unmatched(self):
        seg  = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        pose = _new_pose(bboxes=[[40, 40, 60, 60]], keypoints=[_kps(seed=1)])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64), iou_threshold=0.1)
        assert pts[0].keypoints is None

    def test_iou_at_threshold_is_included(self):
        # identical bbox → IoU = 1.0, always ≥ any threshold ≤ 1
        seg  = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        pose = _new_pose(bboxes=[[0, 0, 10, 10]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64), iou_threshold=0.99)
        assert pts[0].keypoints is not None

    def test_iou_below_threshold_excluded(self):
        # 10×10 track, pose overlaps by 1 pixel → IoU ≈ 0.005
        seg  = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        pose = _new_pose(bboxes=[[9, 9, 19, 19]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64), iou_threshold=0.1)
        assert pts[0].keypoints is None

    def test_two_separated_players_each_matched(self):
        bA, bB = [0, 0, 15, 15], [40, 40, 55, 55]
        kA, kB = _kps(seed=10), _kps(seed=20)
        seg  = _seg(bboxes=[bA, bB], tracker_ids=[1, 2])
        pose = _new_pose(bboxes=[bA, bB], keypoints=[kA, kB])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert pts[0].keypoints is not None
        assert pts[1].keypoints is not None
        np.testing.assert_array_equal(pts[0].keypoints, kA)
        np.testing.assert_array_equal(pts[1].keypoints, kB)

    def test_hungarian_finds_globally_optimal_assignment(self):
        """
        Two tracks with matching bboxes.  The diagonal assignment
        (track_i ↔ pose_i) is globally optimal; each track receives its own
        distinct keypoints.
        """
        bA, bB = [0, 0, 20, 20], [30, 30, 50, 50]
        kA, kB = _kps(seed=31), _kps(seed=32)
        seg  = _seg(bboxes=[bA, bB], tracker_ids=[1, 2])
        pose = _new_pose(bboxes=[bA, bB], keypoints=[kA, kB])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        np.testing.assert_array_equal(pts[0].keypoints, kA)
        np.testing.assert_array_equal(pts[1].keypoints, kB)

    def test_extra_poses_ignored(self):
        seg  = _seg(bboxes=[[0, 0, 15, 15]], tracker_ids=[1])
        pose = _new_pose(
            bboxes=[[0, 0, 15, 15], [40, 40, 55, 55]],
            keypoints=[_kps(seed=1), _kps(seed=2)],
        )
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert len(pts) == 1
        assert pts[0].keypoints is not None

    def test_extra_tracks_get_none_keypoints(self):
        seg  = _seg(bboxes=[[0, 0, 15, 15], [40, 40, 55, 55]], tracker_ids=[1, 2])
        pose = _new_pose(bboxes=[[0, 0, 15, 15]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert len(pts) == 2
        assert sum(1 for p in pts if p.keypoints is not None) == 1
        assert pts[1].keypoints is None

    def test_keypoint_scores_assigned_correctly(self):
        sc  = _scores(val=0.75)
        seg = _seg(bboxes=[[0, 0, 15, 15]], tracker_ids=[1])
        pose = _new_pose(bboxes=[[0, 0, 15, 15]], keypoints=[_kps()], scores=[sc])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        np.testing.assert_array_almost_equal(pts[0].keypoint_scores, sc)

    def test_empty_pose_result_leaves_all_tracks_unmatched(self):
        seg  = _seg(bboxes=[[0, 0, 15, 15], [30, 0, 45, 15]], tracker_ids=[1, 2])
        pose = _new_pose(bboxes=[], keypoints=[])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert all(p.keypoints is None for p in pts)


# ---------------------------------------------------------------------------
# TestIoUAssociationYoloFormat  — legacy ultralytics pose result
# ---------------------------------------------------------------------------

class TestIoUAssociationYoloFormat:
    def test_single_player_matched(self):
        bbox = [0, 0, 20, 20]
        kps  = _kps(seed=5)
        seg  = _seg(bboxes=[bbox], tracker_ids=[1])
        pose = _YoloPose(bboxes=[bbox], keypoints=[kps])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert pts[0].keypoints is not None
        np.testing.assert_array_almost_equal(pts[0].keypoints, kps)

    def test_no_conf_scores_default_to_ones(self):
        bbox = [0, 0, 20, 20]
        seg  = _seg(bboxes=[bbox], tracker_ids=[1])
        pose = _YoloPose(bboxes=[bbox], keypoints=[_kps()], no_conf=True)
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        np.testing.assert_array_almost_equal(
            pts[0].keypoint_scores, np.ones(17, dtype=np.float32)
        )

    def test_no_boxes_bbox_inferred_from_keypoints(self):
        # Keypoints span [5, 10] → [25, 30]; track bbox matches that range.
        kps = np.zeros((17, 2), dtype=np.float32)
        kps[:, 0] = np.linspace(5, 25, 17)
        kps[:, 1] = np.linspace(10, 30, 17)
        seg  = _seg(bboxes=[[5, 10, 25, 30]], tracker_ids=[1])
        pose = _YoloPose(bboxes=[[5, 10, 25, 30]], keypoints=[kps], no_boxes=True)
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert pts[0].keypoints is not None

    def test_two_players_each_matched(self):
        bA, bB = [0, 0, 15, 15], [40, 40, 55, 55]
        kA, kB = _kps(seed=11), _kps(seed=22)
        seg  = _seg(bboxes=[bA, bB], tracker_ids=[1, 2])
        pose = _YoloPose(bboxes=[bA, bB], keypoints=[kA, kB])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert pts[0].keypoints is not None
        assert pts[1].keypoints is not None

    def test_unmatched_yolo_player_has_none_keypoints(self):
        seg  = _seg(bboxes=[[0, 0, 15, 15], [50, 50, 60, 60]], tracker_ids=[1, 2])
        pose = _YoloPose(bboxes=[[0, 0, 15, 15]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert pts[1].keypoints is None


# ---------------------------------------------------------------------------
# TestCenterDistanceFallback
# ---------------------------------------------------------------------------

class TestCenterDistanceFallback:
    def test_no_fallback_when_max_center_dist_is_none(self):
        # Non-overlapping: no IoU match and no distance fallback → unmatched
        seg  = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        pose = _new_pose(bboxes=[[30, 30, 40, 40]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(
            seg, pose, (64, 64), iou_threshold=0.1, max_center_dist=None
        )
        assert pts[0].keypoints is None

    def test_fallback_matches_close_non_overlapping_pose(self):
        # Track center (5,5), pose center (20,5): no IoU but dist=15 < 50
        seg  = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        pose = _new_pose(bboxes=[[15, 0, 25, 10]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(
            seg, pose, (64, 64), iou_threshold=0.1, max_center_dist=50.0
        )
        assert pts[0].keypoints is not None

    def test_fallback_too_far_leaves_unmatched(self):
        # Track center (5,5), pose center (65,5): dist=60 > max_center_dist=30
        seg  = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        pose = _new_pose(bboxes=[[60, 0, 70, 10]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(
            seg, pose, (64, 64), iou_threshold=0.1, max_center_dist=30.0
        )
        assert pts[0].keypoints is None

    def test_fallback_does_not_reuse_iou_matched_pose(self):
        """
        Track 1 wins pose 0 via IoU.  Track 2 is close enough for a distance
        match, but pose 0 is already consumed — so track 2 stays unmatched.
        """
        seg  = _seg(bboxes=[[0, 0, 20, 20], [25, 0, 35, 20]], tracker_ids=[1, 2])
        pose = _new_pose(bboxes=[[0, 0, 20, 20]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(
            seg, pose, (64, 64), iou_threshold=0.1, max_center_dist=200.0
        )
        assert sum(1 for p in pts if p.keypoints is not None) == 1

    def test_fallback_assigns_closer_track_wins(self):
        """
        Two unmatched tracks compete for one fallback pose.
        Track 1 center (5,5), Track 2 center (50,50).
        Pose center (17, 5) — dist to track1 = 12, dist to track2 ≈ 55.
        Only track 1 is within max_center_dist=30 so it wins.
        """
        seg  = _seg(bboxes=[[0, 0, 10, 10], [45, 45, 55, 55]], tracker_ids=[1, 2])
        pose = _new_pose(bboxes=[[12, 0, 22, 10]], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(
            seg, pose, (64, 64), iou_threshold=0.5, max_center_dist=30.0
        )
        matched_ids = {p.id for p in pts if p.keypoints is not None}
        assert matched_ids == {1}


# ---------------------------------------------------------------------------
# TestOutputStructure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_each_result_is_player_track_instance(self):
        seg = _seg(bboxes=[[0, 0, 10, 10], [20, 20, 30, 30]], tracker_ids=[1, 2])
        pts, _ = associate_poses_with_tracks(seg, None, (64, 64))
        assert all(isinstance(p, PlayerTrack) for p in pts)

    def test_player_track_mask_matches_input(self):
        m   = _mask(2, 2, 8, 8)
        seg = _seg(bboxes=[[2, 2, 8, 8]], tracker_ids=[5], masks=[m])
        pts, _ = associate_poses_with_tracks(seg, None, (64, 64))
        np.testing.assert_array_equal(pts[0].mask, m)

    def test_player_track_bbox_matches_input(self):
        bbox = [2, 3, 8, 9]
        seg  = _seg(bboxes=[bbox], tracker_ids=[99])
        pts, _ = associate_poses_with_tracks(seg, None, (64, 64))
        np.testing.assert_array_almost_equal(pts[0].bbox, bbox)

    def test_player_track_id_matches_input(self):
        seg = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[42])
        pts, _ = associate_poses_with_tracks(seg, None, (64, 64))
        assert pts[0].id == 42

    def test_player_tracks_is_list(self):
        seg = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        pts, _ = associate_poses_with_tracks(seg, None, (64, 64))
        assert isinstance(pts, list)

    def test_ball_track_none_when_no_ball_center(self):
        seg = _seg(bboxes=[[0, 0, 10, 10]], tracker_ids=[1])
        _, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert bt is None

    def test_ball_track_is_ball_track_instance(self):
        seg = _seg_with_ball(
            bboxes=[[0, 0, 10, 10]],
            tracker_ids=[-1],
            masks=[_mask(0, 0, 10, 10)],
            ball_center=(5.0, 5.0),
        )
        _, bt = associate_poses_with_tracks(seg, None, (64, 64))
        assert isinstance(bt, BallTrack)

    def test_pose_matched_player_has_non_none_keypoints(self):
        bbox = [0, 0, 20, 20]
        seg  = _seg(bboxes=[bbox], tracker_ids=[7])
        pose = _new_pose(bboxes=[bbox], keypoints=[_kps()])
        pts, _ = associate_poses_with_tracks(seg, pose, (64, 64))
        assert pts[0].keypoints is not None
        assert pts[0].keypoint_scores is not None
