"""
segmentation_tracking package

Provides player instance segmentation, persistent tracking, pose estimation
integration, team classification, and visualization for football video analysis.

Modules:
    segmentation_model  -- BoT-SORT + SAM2 tracker for players and ball
    association         -- Hungarian pose-to-mask matching
    visualization       -- Frame annotation with masks, IDs, and skeletons
    ball_kalman         -- Detection-first + MOSSE DCF ball tracker (BallDCFTracker); FRoG-MOT motion-state; UKF retained for compat
    team_classifier     -- HSV K-means jersey-colour team classifier
    vector_field        -- Player velocity tracking and vector-field attractor estimation
"""

from segmentation_tracking.segmentation_model import SegmentationTracker, SegmentationResult
from segmentation_tracking.association import PlayerTrack, BallTrack, associate_poses_with_tracks
from segmentation_tracking.visualization import Visualizer
from segmentation_tracking.ball_kalman import BallKalmanFilter, AdaptiveBallKalmanFilter, BallDCFTracker, BallMotionState, BallCoTrackerTracker
from segmentation_tracking.team_classifier import TeamClassifier
from segmentation_tracking.vector_field import PlayerVelocityTracker, KeypointVelocityTracker, AttractorEstimate, estimate_attractor, AttractorSmoother

__all__ = [
    "SegmentationTracker",
    "SegmentationResult",
    "PlayerTrack",
    "BallTrack",
    "associate_poses_with_tracks",
    "Visualizer",
    "BallKalmanFilter",
    "AdaptiveBallKalmanFilter",
    "BallDCFTracker",
    "BallCoTrackerTracker",
    "BallMotionState",
    "TeamClassifier",
    "PlayerVelocityTracker",
    "KeypointVelocityTracker",
    "AttractorEstimate",
    "estimate_attractor",
    "AttractorSmoother",
]
