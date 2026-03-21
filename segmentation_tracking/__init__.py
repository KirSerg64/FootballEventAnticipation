"""
segmentation_tracking package

Provides player instance segmentation, persistent tracking, pose estimation
integration, team classification, and visualization for football video analysis.

Modules:
    segmentation_model  -- BoT-SORT + SAM2 tracker for players and ball
    sam3_wrapper        -- SAM3 text-prompt tracker (drop-in SAM2 replacement)
    association         -- Hungarian pose-to-mask matching
    visualization       -- Frame annotation with masks, IDs, and skeletons
    ball_kalman         -- Detection-first + MOSSE DCF ball tracker (BallDCFTracker); FRoG-MOT motion-state; UKF retained for compat
    team_classifier     -- HSV K-means and SIGLIP-embedding team classifiers
    vector_field        -- Player velocity tracking and vector-field attractor estimation
"""

from segmentation_tracking.segmentation_model import SegmentationTracker, SegmentationResult
from segmentation_tracking.sam3_wrapper import Sam3SegmentationTracker
from segmentation_tracking.association import PlayerTrack, BallTrack, associate_poses_with_tracks
from segmentation_tracking.visualization import Visualizer
from segmentation_tracking.ball_kalman import BallKalmanFilter, AdaptiveBallKalmanFilter, BallDCFTracker, BallMotionState, BallCoTrackerTracker
from segmentation_tracking.team_classifier import TeamClassifier, SiglipTeamClassifier, create_team_classifier
from segmentation_tracking.vector_field import PlayerVelocityTracker, KeypointVelocityTracker, AttractorEstimate, estimate_attractor, AttractorSmoother

__all__ = [
    "SegmentationTracker",
    "Sam3SegmentationTracker",
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
    "SiglipTeamClassifier",
    "create_team_classifier",
    "PlayerVelocityTracker",
    "KeypointVelocityTracker",
    "AttractorEstimate",
    "estimate_attractor",
    "AttractorSmoother",
]
