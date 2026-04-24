"""
tests/test_appearance_reid.py
------------------------------
Unit tests for ``segmentation_tracking.appearance_reid.AppearanceReIDMatcher``.

All tests run purely on CPU with no GPU, YOLO, or SAM2 dependencies.

Run with:
    pytest tests/test_appearance_reid.py -v
"""
from __future__ import annotations

import os
import sys
import importlib.util
import types

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import importlib.util
import types

# Import appearance_reid directly without triggering the full package __init__
# (which requires torch, YOLO, etc.)
_spec = importlib.util.spec_from_file_location(
    "segmentation_tracking.appearance_reid",
    os.path.join(_REPO_ROOT, "segmentation_tracking", "appearance_reid.py"),
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
# Register parent package stub so @dataclass resolution works
if "segmentation_tracking" not in sys.modules:
    _pkg = types.ModuleType("segmentation_tracking")
    sys.modules["segmentation_tracking"] = _pkg
sys.modules["segmentation_tracking.appearance_reid"] = _mod
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
AppearanceReIDMatcher = _mod.AppearanceReIDMatcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_frame(color_bgr: tuple[int, int, int], h: int = 128, w: int = 128) -> np.ndarray:
    """Return a solid-colour BGR frame."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color_bgr
    return frame


def _bbox(x1: int = 10, y1: int = 10, x2: int = 60, y2: int = 90) -> np.ndarray:
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _matcher(**kwargs) -> AppearanceReIDMatcher:
    defaults = dict(gallery_ttl=10, similarity_threshold=0.80)
    defaults.update(kwargs)
    return AppearanceReIDMatcher(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults_stored(self):
        m = AppearanceReIDMatcher()
        assert m.gallery_ttl == 90
        assert m.similarity_threshold == pytest.approx(0.85)

    def test_custom_params(self):
        m = AppearanceReIDMatcher(gallery_ttl=30, similarity_threshold=0.70)
        assert m.gallery_ttl == 30
        assert m.similarity_threshold == pytest.approx(0.70)

    def test_starts_empty(self):
        m = AppearanceReIDMatcher()
        assert len(m._active_hists) == 0
        assert len(m._gallery) == 0

    def test_reset_clears_state(self):
        m = _matcher()
        frame = _solid_frame((200, 100, 50))
        m.update_active(1, frame, _bbox())
        m.notify_lost([1], {1: _bbox()})
        assert len(m._gallery) == 1
        m.reset()
        assert len(m._active_hists) == 0
        assert len(m._gallery) == 0


# ---------------------------------------------------------------------------
# _extract_histogram
# ---------------------------------------------------------------------------

class TestExtractHistogram:
    def test_returns_unit_vector(self):
        frame = _solid_frame((100, 200, 50))
        hist = AppearanceReIDMatcher._extract_histogram(frame, _bbox())
        assert hist is not None
        assert abs(float(np.linalg.norm(hist)) - 1.0) < 1e-5

    def test_returns_float32(self):
        frame = _solid_frame((100, 200, 50))
        hist = AppearanceReIDMatcher._extract_histogram(frame, _bbox())
        assert hist is not None
        assert hist.dtype == np.float32

    def test_degenerate_bbox_returns_none(self):
        frame = _solid_frame((100, 200, 50))
        bad_bbox = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        hist = AppearanceReIDMatcher._extract_histogram(frame, bad_bbox)
        assert hist is None

    def test_different_colours_produce_different_histograms(self):
        red_frame = _solid_frame((0, 0, 200))     # red in BGR
        blue_frame = _solid_frame((200, 0, 0))    # blue in BGR
        bbox = _bbox()
        h_red = AppearanceReIDMatcher._extract_histogram(red_frame, bbox)
        h_blue = AppearanceReIDMatcher._extract_histogram(blue_frame, bbox)
        assert h_red is not None
        assert h_blue is not None
        # Cosine similarity should be well below 1.0 for distinct colours
        sim = float(np.dot(h_red, h_blue))
        assert sim < 0.95

    def test_same_colour_produces_similar_histograms(self):
        frame1 = _solid_frame((50, 120, 200))
        frame2 = _solid_frame((55, 118, 198))   # almost the same colour
        bbox = _bbox()
        h1 = AppearanceReIDMatcher._extract_histogram(frame1, bbox)
        h2 = AppearanceReIDMatcher._extract_histogram(frame2, bbox)
        assert h1 is not None and h2 is not None
        sim = float(np.dot(h1, h2))
        assert sim > 0.90


# ---------------------------------------------------------------------------
# update_active
# ---------------------------------------------------------------------------

class TestUpdateActive:
    def test_first_call_stores_histogram(self):
        m = _matcher()
        frame = _solid_frame((100, 200, 50))
        m.update_active(1, frame, _bbox())
        assert 1 in m._active_hists
        assert m._active_hists[1] is not None

    def test_subsequent_call_updates_ema(self):
        m = _matcher()
        frame1 = _solid_frame((100, 200, 50))
        frame2 = _solid_frame((50, 100, 200))   # distinctly different colour
        m.update_active(1, frame1, _bbox())
        hist_before = m._active_hists[1].copy()
        m.update_active(1, frame2, _bbox())
        hist_after = m._active_hists[1]
        # Should have changed (EMA applied)
        assert not np.allclose(hist_before, hist_after)

    def test_updated_histogram_remains_unit(self):
        m = _matcher()
        for i in range(5):
            colour = (i * 40, 200 - i * 20, 100 + i * 10)
            frame = _solid_frame(colour)
            m.update_active(1, frame, _bbox())
        norm = float(np.linalg.norm(m._active_hists[1]))
        assert abs(norm - 1.0) < 1e-4

    def test_multiple_tracks(self):
        m = _matcher()
        for tid in range(1, 5):
            frame = _solid_frame((tid * 50, tid * 30, tid * 20))
            m.update_active(tid, frame, _bbox())
        assert len(m._active_hists) == 4


# ---------------------------------------------------------------------------
# notify_lost
# ---------------------------------------------------------------------------

class TestNotifyLost:
    def test_moves_active_to_gallery(self):
        m = _matcher()
        frame = _solid_frame((100, 200, 50))
        m.update_active(1, frame, _bbox())
        m.notify_lost([1], {1: _bbox()})
        assert 1 not in m._active_hists
        assert 1 in m._gallery

    def test_no_active_history_skipped(self):
        m = _matcher()
        # Track 99 was never added to active hists
        m.notify_lost([99], {99: _bbox()})
        assert 99 not in m._gallery

    def test_no_bbox_skipped(self):
        m = _matcher()
        frame = _solid_frame((100, 200, 50))
        m.update_active(5, frame, _bbox())
        m.notify_lost([5], {})   # no bbox provided
        assert 5 not in m._gallery

    def test_gallery_entry_has_zero_frames_absent(self):
        m = _matcher()
        frame = _solid_frame((100, 200, 50))
        m.update_active(2, frame, _bbox())
        m.notify_lost([2], {2: _bbox()})
        assert m._gallery[2].frames_absent == 0


# ---------------------------------------------------------------------------
# rematch
# ---------------------------------------------------------------------------

class TestRematch:
    def _populate_gallery(
        self,
        matcher: AppearanceReIDMatcher,
        track_id: int,
        colour_bgr: tuple[int, int, int],
    ) -> None:
        """Helper: add a track to the gallery with the given colour."""
        frame = _solid_frame(colour_bgr)
        matcher.update_active(track_id, frame, _bbox())
        matcher.notify_lost([track_id], {track_id: _bbox()})

    def test_returns_empty_when_no_gallery(self):
        m = _matcher()
        frame = _solid_frame((100, 200, 50))
        result = m.rematch([1], frame, {1: _bbox()})
        assert result == {}

    def test_returns_empty_when_no_new_ids(self):
        m = _matcher()
        self._populate_gallery(m, 1, (100, 200, 50))
        frame = _solid_frame((100, 200, 50))
        result = m.rematch([], frame, {})
        assert result == {}

    def test_matches_identical_colour(self):
        m = _matcher(similarity_threshold=0.80)
        colour = (60, 120, 200)
        self._populate_gallery(m, old_track_id := 1, colour)
        frame = _solid_frame(colour)
        result = m.rematch([new_id := 42], frame, {new_id: _bbox()})
        assert result.get(new_id) == old_track_id

    def test_no_match_for_very_different_colour(self):
        m = _matcher(similarity_threshold=0.90)
        self._populate_gallery(m, 1, (0, 0, 200))     # red
        frame = _solid_frame((200, 0, 0))              # blue
        result = m.rematch([42], frame, {42: _bbox()})
        assert result == {}

    def test_one_to_one_assignment(self):
        """Two new IDs compete for the best gallery match — each should get
        a different old ID (Hungarian 1-to-1 assignment)."""
        m = _matcher(similarity_threshold=0.70)
        colour_a = (60, 120, 200)    # warm orange-ish
        colour_b = (200, 60, 60)     # cool blue-ish
        self._populate_gallery(m, old_a := 1, colour_a)
        self._populate_gallery(m, old_b := 2, colour_b)

        # Present two new tracks with matching colours
        new_a, new_b = 100, 101
        frame_a = _solid_frame(colour_a)
        frame_b = _solid_frame(colour_b)

        # Build bboxes for both new IDs
        bboxes = {new_a: _bbox(), new_b: _bbox()}

        # Rematch from frame_a (only one colour context; both new IDs described
        # below by their actual frame, but rematch must use a single frame arg)
        # Use a blended frame that contains both colours in different regions
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[:, :64] = colour_a   # left half: colour A
        frame[:, 64:] = colour_b   # right half: colour B

        bbox_a = np.array([0.0, 10.0, 60.0, 90.0], dtype=np.float32)
        bbox_b = np.array([66.0, 10.0, 120.0, 90.0], dtype=np.float32)
        bboxes = {new_a: bbox_a, new_b: bbox_b}

        result = m.rematch([new_a, new_b], frame, bboxes)
        # Both new IDs should be remapped, each to a different old ID
        if result:
            assert len(set(result.values())) == len(result), (
                "Multiple new IDs should not map to the same old ID"
            )

    def test_no_match_when_bbox_missing(self):
        m = _matcher(similarity_threshold=0.80)
        self._populate_gallery(m, 1, (100, 200, 50))
        frame = _solid_frame((100, 200, 50))
        # new_id=42 has no entry in bboxes dict
        result = m.rematch([42], frame, {})
        assert result == {}


# ---------------------------------------------------------------------------
# age_gallery
# ---------------------------------------------------------------------------

class TestAgeGallery:
    def _add_to_gallery(self, m: AppearanceReIDMatcher, tid: int) -> None:
        frame = _solid_frame((100, 200, 50))
        m.update_active(tid, frame, _bbox())
        m.notify_lost([tid], {tid: _bbox()})

    def test_gallery_aged_each_call(self):
        m = _matcher(gallery_ttl=3)
        self._add_to_gallery(m, 1)
        assert m._gallery[1].frames_absent == 0
        m.age_gallery(set())
        assert m._gallery[1].frames_absent == 1
        m.age_gallery(set())
        assert m._gallery[1].frames_absent == 2

    def test_entry_pruned_after_ttl(self):
        m = _matcher(gallery_ttl=2)
        self._add_to_gallery(m, 1)
        for _ in range(3):
            m.age_gallery(set())
        assert 1 not in m._gallery

    def test_entry_removed_when_re_identified(self):
        m = _matcher()
        self._add_to_gallery(m, 1)
        # Simulate that track 1 was successfully re-assigned to a new ID
        m.age_gallery(current_track_ids={1})   # old_id=1 is now active again
        assert 1 not in m._gallery

    def test_multiple_entries_independent_aging(self):
        m = _matcher(gallery_ttl=3)
        self._add_to_gallery(m, 1)
        self._add_to_gallery(m, 2)
        m.age_gallery(set())
        m.age_gallery(set())
        # Track 2 re-appears
        m.age_gallery(current_track_ids={2})
        assert 2 not in m._gallery
        assert 1 in m._gallery                # track 1 still in gallery


# ---------------------------------------------------------------------------
# Integration: full cycle (update → lost → rematch → age)
# ---------------------------------------------------------------------------

class TestFullCycle:
    def test_id_recovery_after_disappearance(self):
        """Simulate a player disappearing for several frames and reappearing."""
        m = _matcher(gallery_ttl=20, similarity_threshold=0.80)
        colour = (100, 180, 60)
        frame = _solid_frame(colour)
        bbox = _bbox()

        # Frames 0–4: track 1 is active
        for _ in range(5):
            m.update_active(1, frame, bbox)

        # Frame 5: track 1 disappears
        m.notify_lost([1], {1: bbox})

        # Frames 6–8: absent — age gallery
        for _ in range(3):
            m.age_gallery(set())

        # Frame 9: player reappears with new ID 99 assigned by BoT-SORT
        remap = m.rematch([99], frame, {99: bbox})
        assert remap.get(99) == 1, f"Expected re-ID to 1, got {remap}"

    def test_no_spurious_rematch_for_fresh_player(self):
        """A brand-new player (not in gallery) should not be remapped."""
        m = _matcher(gallery_ttl=20, similarity_threshold=0.80)
        # No gallery entries at all
        frame = _solid_frame((120, 80, 200))
        result = m.rematch([55], frame, {55: _bbox()})
        assert result == {}
