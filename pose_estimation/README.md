# Pose Estimation Module — Football Event Anticipation

Experimental module for evaluating player pose estimation in football video.

---

## Overview

This module adds a **pose estimation pipeline** to the Football Event Anticipation project.  
It is **exploratory only** and does not modify the existing training pipeline.

### Supported backends

| Backend | Notes |
|---|---|
| `keypointrcnn` (default) | torchvision ResNet-50 FPN — no extra install required |
| `bboxmaskpose` | [BBoxMaskPose](https://github.com/MiraPurkrabek/BBoxMaskPose) — better accuracy for occluded players |

---

## Directory structure

```
pose_estimation/
    __init__.py          Package init
    pose_model.py        Model loading, inference abstraction
    pose_inference.py    Frame/video iteration pipeline
    visualization.py     Skeleton drawing, video writing
    README.md            This file
scripts/
    run_pose_estimation.py   CLI entry point
output/                 Generated output videos (git-ignored)
```

---

## Installation

### 1. Core dependencies (required)

```bash
pip install torch torchvision opencv-python numpy
```

> Tested with Python 3.10+, PyTorch ≥ 2.0, torchvision ≥ 0.15.

A `requirements_pose.txt` is provided at the repository root for convenience:

```bash
pip install -r requirements_pose.txt
```

### 2. BBoxMaskPose (optional — for higher accuracy)

BBoxMaskPose is a ViTPose-based model adapted for occluded football players.

```bash
# Install MMPose ecosystem
pip install openmim
mim install mmcv mmdet mmpose

# Clone and install BBoxMaskPose
git clone https://github.com/MiraPurkrabek/BBoxMaskPose.git
cd BBoxMaskPose && pip install -e .
```

Pretrained weights are available from the BBoxMaskPose repository releases page.

---

## Usage

### Minimal (default KeypointRCNN backend)

```bash
python scripts/run_pose_estimation.py \
    --input path/to/video.mp4 \
    --output output/pose_visualization.mp4
```

### With GPU

```bash
python scripts/run_pose_estimation.py \
    --input path/to/video.mp4 \
    --output output/pose_visualization.mp4 \
    --device cuda
```

### Directory of frames

```bash
python scripts/run_pose_estimation.py \
    --input path/to/frames_dir/ \
    --output output/pose_visualization.mp4 \
    --device cuda
```

### With BBoxMaskPose

```bash
python scripts/run_pose_estimation.py \
    --input path/to/video.mp4 \
    --output output/pose_visualization.mp4 \
    --device cuda \
    --backend bboxmaskpose \
    --bbox_config BBoxMaskPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    --bbox_checkpoint BBoxMaskPose/weights/vitpose_base.pth \
    --det_config BBoxMaskPose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    --det_checkpoint https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/...pth
```

### All options

```
usage: run_pose_estimation.py [-h]
    --input INPUT
    [--output OUTPUT]
    [--device DEVICE]
    [--backend {auto,keypointrcnn,bboxmaskpose}]
    [--weights WEIGHTS]
    [--bbox_config BBOX_CONFIG]
    [--bbox_checkpoint BBOX_CHECKPOINT]
    [--det_config DET_CONFIG]
    [--det_checkpoint DET_CHECKPOINT]
    [--max_frames MAX_FRAMES]
    [--confidence_threshold CONFIDENCE_THRESHOLD]
    [--show_bbox | --hide_bbox]
    [--show_confidence]
    [--verbose]
```

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | Video file or frames directory |
| `--output` | `output/pose_visualization.mp4` | Output video path |
| `--device` | `auto` | `auto`, `cuda`, `cuda:0`, `cpu` |
| `--backend` | `auto` | `auto`, `keypointrcnn`, `bboxmaskpose` |
| `--max_frames` | all | Limit frames processed |
| `--confidence_threshold` | `0.5` | Min score for detections |
| `--show_confidence` | off | Overlay keypoint scores |
| `--verbose` | off | Debug logging |

---

## Output

The output video (`output/pose_visualization.mp4`) shows:

- **Orange bounding boxes** around detected players
- **Coloured skeleton limbs** connecting joints
- **Cyan keypoints** at each joint location
- Same resolution and frame order as the input

---

## Programmatic API

```python
import torch
from pose_estimation.pose_model import create_pose_estimator
from pose_estimation.pose_inference import PoseInferencePipeline, get_video_properties
from pose_estimation.visualization import PoseVisualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

estimator = create_pose_estimator(device=device)
pipeline = PoseInferencePipeline(estimator, max_frames=100)
vis = PoseVisualizer("output/pose_visualization.mp4")

w, h, fps = get_video_properties("video.mp4")
vis.open(w, h, fps)
for _, frame, pose_result in pipeline.run("video.mp4"):
    vis.process_frame(frame, pose_result)
vis.close()
```

---

## Notes

- The `output/` directory is created automatically if it does not exist.
- The default **KeypointRCNN** backend downloads COCO-pretrained weights (~150 MB) from
  the torchvision model zoo on first run.
- **BBoxMaskPose** requires additional model weights to be downloaded separately
  (see the [BBoxMaskPose repository](https://github.com/MiraPurkrabek/BBoxMaskPose)).
- Both backends produce COCO-format 17-keypoint poses.
