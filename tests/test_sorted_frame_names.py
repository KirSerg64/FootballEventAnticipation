#!/usr/bin/env python3
"""
Tests for _sorted_frame_names() in optical_flow/process_optical_flow.py.

Uses real clip directories from data/soccernetballanticipation/720p:
  - clip_1   contains only .mp4 files (no extracted frames)
  - clip_10  contains frame1.jpg … frame750.jpg
"""

import os
import sys
import tempfile
from loguru import logger

import pytest

# ---------------------------------------------------------------------------
# Bring optical_flow/ onto sys.path so the import below works from any cwd.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPT_FLOW_DIR = os.path.join(_REPO_ROOT, "optical_flow")
if _OPT_FLOW_DIR not in sys.path:
    sys.path.insert(0, _OPT_FLOW_DIR)

from process_optical_flow import _sorted_frame_names  # noqa: E402

# ---------------------------------------------------------------------------
# Paths to real clip directories
# ---------------------------------------------------------------------------
_CLIPS_ROOT = os.path.join(
    _REPO_ROOT, "data", "soccernetballanticipation", "720p", "train"
)
_CLIP_NO_FRAMES = os.path.join(_CLIPS_ROOT, "clip_1")   # only .mp4 files
_CLIP_WITH_FRAMES = os.path.join(_CLIPS_ROOT, "clip_10")  # frame1…frame750


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _frame_number(name: str) -> int:
    """Extract the integer from 'frame<N>.jpg'."""
    return int(name[len("frame"):name.rfind(".")])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSortedFrameNamesRealData:

    def test_clip_with_no_frames_returns_empty_list(self):
        """clip_1 has only .mp4 files — function must return []."""
        result = _sorted_frame_names(_CLIP_NO_FRAMES)
        assert result == [], (
            f"Expected empty list for clip with no frames, got {result[:5]}"
        )

    def test_clip_with_frames_returns_non_empty_list(self):
        """clip_10 has 750 extracted JPEG frames."""
        result = _sorted_frame_names(_CLIP_WITH_FRAMES)
        print(result)
        assert len(result) > 0

    def test_correct_frame_count(self):
        """clip_10 should have exactly 750 frames."""
        result = _sorted_frame_names(_CLIP_WITH_FRAMES)
        assert len(result) == 750

    def test_sorted_ascending_by_frame_number(self):
        """Result must be sorted strictly ascending by the numeric index."""
        result = _sorted_frame_names(_CLIP_WITH_FRAMES)
        numbers = [_frame_number(n) for n in result]
        assert numbers == sorted(numbers), "Frame names are not in ascending numeric order"

    def test_first_frame_is_frame1(self):
        """Frames are 1-indexed; the first entry must be frame1.jpg."""
        result = _sorted_frame_names(_CLIP_WITH_FRAMES)
        assert result[0] == "frame1.jpg"

    def test_last_frame_is_frame750(self):
        """Last frame in clip_10 is frame750.jpg."""
        result = _sorted_frame_names(_CLIP_WITH_FRAMES)
        assert result[-1] == "frame750.jpg"

    def test_no_gaps_in_sequence(self):
        """Frame numbers must form a contiguous range 1…750 with no gaps."""
        result = _sorted_frame_names(_CLIP_WITH_FRAMES)
        numbers = [_frame_number(n) for n in result]
        assert numbers == list(range(numbers[0], numbers[0] + len(numbers)))

    def test_all_entries_are_jpg(self):
        """Only .jpg / .jpeg / .png files matching frame<N>.<ext> are included."""
        result = _sorted_frame_names(_CLIP_WITH_FRAMES)
        logger.info(result)
        for name in result:
            assert name.lower().endswith((".jpg", ".jpeg", ".png")), (
                f"Unexpected file extension in result: {name}"
            )

    def test_non_frame_files_are_excluded(self):
        """Files that don't match frame<N>.<ext> must not appear in the result."""
        result = _sorted_frame_names(_CLIP_WITH_FRAMES)
        result_set = set(result)
        for name in result:
            assert not name.endswith(".mp4"), f".mp4 file leaked into result: {name}"
            assert not name.endswith(".json"), f".json file leaked into result: {name}"
        # Also verify by checking the raw directory listing
        all_names = set(os.listdir(_CLIP_WITH_FRAMES))
        non_frame = all_names - result_set
        for name in non_frame:
            # Everything excluded must NOT look like frame<N>.<ext>
            import re
            assert not re.match(r"^frame\d+\.(jpg|jpeg|png)$", name, re.IGNORECASE), (
                f"Frame file incorrectly excluded: {name}"
            )


class TestSortedFrameNamesEdgeCases:

    def test_nonexistent_directory_returns_empty_list(self):
        """OSError on a missing path must be swallowed and return []."""
        result = _sorted_frame_names("/nonexistent/path/that/does/not/exist")
        assert result == []

    def test_empty_directory_returns_empty_list(self):
        """A real but empty directory must return []."""
        with tempfile.TemporaryDirectory() as tmp:
            result = _sorted_frame_names(tmp)
            assert result == []

    def test_mixed_extensions_included(self):
        """frame<N>.png and frame<N>.jpeg are also valid frame names."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("frame1.jpg", "frame2.jpeg", "frame3.png",
                         "frame4.mp4", "labels.json", "noframe.jpg"):
                open(os.path.join(tmp, name), "w").close()
            result = _sorted_frame_names(tmp)
            assert result == ["frame1.jpg", "frame2.jpeg", "frame3.png"]

    def test_non_sequential_numbers_still_sorted(self):
        """Numeric sort must hold even when frame numbers are non-contiguous."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("frame10.jpg", "frame2.jpg", "frame100.jpg", "frame1.jpg"):
                open(os.path.join(tmp, name), "w").close()
            result = _sorted_frame_names(tmp)
            numbers = [_frame_number(n) for n in result]
            assert numbers == [1, 2, 10, 100]

    def test_case_insensitive_extension_matching(self):
        """frame1.JPG and frame2.JPEG must be included (case-insensitive)."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("frame1.JPG", "frame2.JPEG", "frame3.PNG"):
                open(os.path.join(tmp, name), "w").close()
            result = _sorted_frame_names(tmp)
            assert len(result) == 3

    def test_files_without_frame_prefix_excluded(self):
        """Files named '0001.jpg' or 'img1.jpg' must not be included."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("frame1.jpg", "0001.jpg", "img1.jpg", "frame.jpg"):
                open(os.path.join(tmp, name), "w").close()
            result = _sorted_frame_names(tmp)
            assert result == ["frame1.jpg"]
