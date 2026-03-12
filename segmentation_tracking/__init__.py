"""
segmentation_tracking package

Provides player instance segmentation, persistent tracking, pose estimation
integration, and visualization for football video analysis.

Modules:
    segmentation_model  – SAM2VideoPredictor-based tracker for players and ball
    association         – Pose-keypoint to segmentation-mask matching
    visualization       – Frame annotation with masks, IDs, and skeletons
"""

from segmentation_tracking.segmentation_model import SegmentationTracker, SegmentationResult
from segmentation_tracking.association import PlayerTrack, associate_poses_with_tracks
from segmentation_tracking.visualization import Visualizer

__all__ = [
    "SegmentationTracker",
    "SegmentationResult",
    "PlayerTrack",
    "associate_poses_with_tracks",
    "Visualizer",
]
