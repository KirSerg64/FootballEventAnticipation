"""
segmentation_tracking package

Provides player instance segmentation, persistent tracking, pose estimation
integration, team classification, and visualization for football video analysis.

Modules:
    segmentation_model  -- BoT-SORT + SAM2 tracker for players and ball
    association         -- Hungarian pose-to-mask matching
    visualization       -- Frame annotation with masks, IDs, and skeletons
    ball_kalman         -- UKF with Laplacian-robust M-estimator for ball centre tracking
    team_classifier     -- HSV K-means jersey-colour team classifier
"""

from segmentation_tracking.segmentation_model import SegmentationTracker, SegmentationResult
from segmentation_tracking.association import PlayerTrack, BallTrack, associate_poses_with_tracks
from segmentation_tracking.visualization import Visualizer
from segmentation_tracking.ball_kalman import BallKalmanFilter, AdaptiveBallKalmanFilter
from segmentation_tracking.team_classifier import TeamClassifier

__all__ = [
    "SegmentationTracker",
    "SegmentationResult",
    "PlayerTrack",
    "BallTrack",
    "associate_poses_with_tracks",
    "Visualizer",
    "BallKalmanFilter",
    "AdaptiveBallKalmanFilter",
    "TeamClassifier",
]
