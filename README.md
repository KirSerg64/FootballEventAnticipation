# FootballEventAnticipation
Football Action Anticipation

---

## Player Segmentation, Tracking & Pose Estimation Pipeline

This module provides a **frame-by-frame analysis pipeline** that combines:

- **Instance segmentation** of players and ball using
  [SAM2VideoPredictor](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/sam/predict.py)
  from the [ultralytics](https://github.com/ultralytics/ultralytics) library.
- **Persistent player IDs** across frames via SAM2's memory-based propagation,
  with YOLO-based re-detection to handle players entering mid-scene.
- **Pose keypoints** (COCO 17-keypoint format) estimated by a YOLO pose model
  and associated with each tracked player via IoU matching.
- **Ball segmentation and position** tracked with YOLO detection on every frame.

### Processing Pipeline

```
Video input
     │
Frame extraction
     │
Player and ball segmentation (SAM2VideoPredictor)
     │
Tracking across frames (SAM2VideoPredictor memory propagation)
     │
Pose estimation (YOLO pose)
     │
Association (mask ↔ keypoints, IoU matching)
     │
Visualization (masks + IDs + skeletons + ball)
     │
Output video: output/segmentation_pose_tracking.mp4
```

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
python scripts/run_segmentation_pose_tracking.py \
    --input path/to/football_video.mp4 \
    --output output/segmentation_pose_tracking.mp4 \
    --device cuda
```

With JSON data export:

```bash
python scripts/run_segmentation_pose_tracking.py \
    --input path/to/football_video.mp4 \
    --output output/segmentation_pose_tracking.mp4 \
    --device cuda \
    --export_json
```

This writes:
- `output/player_tracks.json` – per-frame player IDs, bboxes, masks, keypoints
- `output/ball_track.json` – per-frame ball centre position

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | *(required)* | Path to input video |
| `--output` | `output/segmentation_pose_tracking.mp4` | Output video path |
| `--device` | `cuda` | Torch device (`cuda` or `cpu`) |
| `--sam_model` | `sam2.1_b.pt` | SAM2 model weights |
| `--det_model` | `yolo11x.pt` | YOLO detection model |
| `--pose_model` | `yolo11x-pose.pt` | YOLO pose model |
| `--max_frames` | all | Maximum frames to process |
| `--conf` | `0.25` | Detection confidence threshold |
| `--iou` | `0.5` | IoU threshold for ID matching |
| `--mask_alpha` | `0.40` | Segmentation overlay opacity |
| `--no_skeleton` | off | Disable skeleton rendering |
| `--no_ball` | off | Disable ball overlay |
| `--export_json` | off | Export JSON results |
| `--redetect_interval` | `30` | Re-run YOLO every N frames |

### Code Structure

```
segmentation_tracking/
    __init__.py           – Public package API
    segmentation_model.py – SAM2VideoPredictor tracker + YOLO ball detection
    association.py        – Pose–mask IoU matching; PlayerTrack / BallTrack types
    visualization.py      – Frame annotation (masks, IDs, skeletons, ball)
scripts/
    run_segmentation_pose_tracking.py  – Full pipeline CLI
requirements.txt
```

### Output Format

**Per-player** (`player_tracks.json`):
```json
{
  "frame_index": 0,
  "player_id": 3,
  "bbox": [x1, y1, x2, y2],
  "mask": [[...], ...],
  "keypoints": [[x, y], ...],
  "keypoint_scores": [0.9, ...]
}
```

**Ball** (`ball_track.json`):
```json
{
  "frame_index": 0,
  "ball_center": [cx, cy],
  "bbox": [x1, y1, x2, y2]
}
```
