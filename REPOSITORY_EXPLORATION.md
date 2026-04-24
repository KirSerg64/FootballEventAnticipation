# Football Event Anticipation Repository - Complete Exploration

## Repository Overview

The **FootballEventAnticipation** repository contains implementation code for **FAANTRA** (Football Action ANticipation TRAnsformer), a transformer-based model for predicting future actions in football broadcast videos. This is the official implementation accompanying the CVPR 2025 Workshops paper "Action Anticipation from SoccerNet Football Video Broadcasts".

**Repository:** https://github.com/KirSerg64/FootballEventAnticipation

---

## Directory Structure

```
FootballEventAnticipation/
├── main.py                              # Main training entry point
├── train.py                             # Training loop implementation
├── train_dual.py                        # Joint training with auxiliary tasks
├── test.py                              # Model evaluation script
├── eval.py                              # Evaluation metrics for BAS dataset
├── eval_BAA.py                          # Evaluation metrics for BAA dataset
├── opts.py                              # Configuration argument parser
├── utils.py                             # Utility functions (loss, matching, seeding)
├── visualize_clip_actions.py            # Visualization tool for actions
├── update_split_json.py                 # Dataset split management
├── shard_and_compress_data.py           # Data compression utility
│
├── setup_dataset_BAS.py                 # Ball Action Spotting dataset setup
├── setup_dataset_BAA.py                 # Ball Action Anticipation dataset setup
├── setup_dataset_BAA_sample.py          # BAA sample dataset setup
│
├── model/                               # Model architectures
│   ├── __init__.py
│   ├── futr.py                          # FUTR model implementation (223 lines)
│   ├── extras/
│   │   ├── position.py                  # Positional encoding
│   │   └── transformer.py               # Transformer encoder/decoder (287 lines)
│   └── T_Deed_Modules/                  # Temporal enhancement modules
│       ├── shift.py                     # Temporal shift operations
│       ├── modules.py                   # EDSGP mixer layers
│       └── impl/
│           ├── gsm.py                   # Global-Shift Module
│           └── gsf.py                   # Global-Shift Filter
│
├── dataset/                             # Data loading
│   ├── datasets.py                      # Dataset loader factory (179 lines)
│   └── frame.py                         # Frame dataset classes (979 lines)
│
├── util/                                # Utilities
│   ├── io.py                            # JSON loading utilities
│   └── dataset.py                       # Class loading utilities
│
├── config/                              # Configuration files
│   ├── README.md                        # Config documentation
│   └── SoccerNetBall/
│       ├── Base-Config-BAS.json         # Ball Action Spotting config
│       ├── Base-Config-BAA.json         # Ball Action Anticipation config
│       ├── Base-Config-BAA_Test01.json  # BAA test config
│       └── Base-Config-Joint.json       # Joint training config
│
├── data/                                # Dataset metadata
│   ├── soccernet/                       # SoccerNet v2 data
│   ├── soccernetball/                   # Ball Action Spotting data
│   └── soccernetballanticipation/       # Ball Action Anticipation data
│       ├── train.json, val.json, test.json, challenge.json
│       ├── class.txt
│       └── 720p/train/Labels-ball.json
│
├── assets/                              # Documentation images
│   ├── modelArchitecture.png
│   └── modelArchitecture-white.png
│
├── README.md                            # Main project documentation
├── ChallengeRules.md                    # Challenge submission rules
├── requirements.txt                     # Python dependencies
└── .gitignore

```

---

## Key Branches

1. **`train`** (Default) - Main production code with full FAANTRA implementation
   - Complete training/evaluation pipeline
   - Action anticipation and segmentation models
   - Dataset loaders for multiple datasets
   
2. **`copilot/pose-estimation-experiment`** - Experimental pose estimation branch
   - Player pose estimation pipeline
   - Integration with torchvision KeypointRCNN and BBoxMaskPose
   - Skeleton visualization utilities

3. **`copilot/add-instance-segmentation-tracking`** - Instance segmentation branch
   - Work in progress for instance-level tracking

---

## Core Components

### 1. Model Architecture (FUTR - Future Transformer)

**File:** `model/futr.py` (223 lines)

The FUTR model combines:
- **Backbone:** RegNetY architectures (rny002, rny004, rny006, rny008) from TIMM
- **Temporal Processing:** Optional T-Deed modules (EDSGP-Mixer)
- **Transformer:** 
  - Encoder: Processes observation frames
  - Decoder: Generates action queries for anticipation
- **Multiple Output Heads:**
  - Action classification head (`fc`)
  - Temporal offset head (`fc_len`)
  - Actionness head (`fc_actionness`) - optional
  - Segmentation head (`fc_seg`) - for auxiliary task

**Key Features:**
- Handles variable observation percentages (obs_perc)
- Supports attention masking for local context
- Joint training with auxiliary segmentation task
- Dual optimizer support (AdamW + Muon)

### 2. Dataset Loading

**File:** `dataset/frame.py` (979 lines), `dataset/datasets.py` (179 lines)

Three supported datasets:
- **SoccerNet v2** - General action spotting
- **Ball Action Spotting (BAS)** - Ball-centric events
- **Ball Action Anticipation (BAA)** - Anticipation-specific dataset

**Key Classes:**
- `ActionSpotDataset` - Main training/validation dataset
- `ActionSpotVideoDataset` - Video-level evaluation dataset
- `ActionSpotDatasetJoint` - Multi-dataset joint training

Features:
- Clip-based sampling with configurable stride/overlap
- Multiple observation percentages support
- Label smoothing and class weighting
- Temporal offset ground truth for actions
- Padding and masking mechanisms

### 3. Training & Evaluation

**Files:** 
- `main.py` - Main training orchestrator (234 lines)
- `train.py` - Training loop with mAP calculation
- `train_dual.py` - Joint training with segmentation
- `test.py` - Evaluation script
- `eval.py` - Evaluation for BAS dataset
- `eval_BAA.py` - Evaluation for BAA dataset

**Key Metrics:**
- mAP@δ - Temporal precision (within δ seconds)
- mAP@∞ - Occurrence within anticipation window
- F1 scores for frame-level predictions
- Class-wise statistics (TP, FP, FN, TN)

---

## Dependencies

### Main Requirements (`requirements.txt`)
```
numpy>=1.26.4
scipy>=1.15.3
torch>=2.0.0 (tested with 2.8.0a0)
einops>=0.8.1
timm>=1.0.19
SoccerNet>=0.1.62
wandb>=0.21.1
tabulate>=0.9.0
pyzipper>=0.3.6
```

### Pose Estimation Branch (`requirements_pose.txt`)
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.24.0

# Optional: For BBoxMaskPose backend
# mmcv, mmdet, mmpose
```

---

## Pose Estimation Module (copilot/pose-estimation-experiment)

### Files:
- `pose_estimation/pose_model.py` - Model abstraction and backends
- `pose_estimation/pose_inference.py` - Frame/video processing pipeline
- `pose_estimation/visualization.py` - Skeleton drawing and video output
- `scripts/run_pose_estimation.py` - CLI entry point

### Features:
- **Two backends:**
  1. **KeypointRCNN** (default) - torchvision ResNet-50 FPN, COCO-pretrained
  2. **BBoxMaskPose** (optional) - ViTPose with bounding-box masking for occluded players

- **Inputs:** Video files or directories of images
- **Output:** Annotated MP4 with skeleton overlays

- **COCO 17-keypoint skeleton:**
  - Nose, eyes, ears (5 points)
  - Shoulders, elbows, wrists (6 points)
  - Hips, knees, ankles (6 points)

### Example Usage:
```bash
# Basic usage
python scripts/run_pose_estimation.py \
    --input video.mp4 \
    --output output.mp4 \
    --device cuda

# With BBoxMaskPose
python scripts/run_pose_estimation.py \
    --input video.mp4 \
    --output output.mp4 \
    --backend bboxmaskpose \
    --bbox_config config.py \
    --bbox_checkpoint weights.pth \
    --det_config det_config.py \
    --det_checkpoint det_weights.pth
```

---

## Configuration System

**Location:** `config/SoccerNetBall/`

Each config JSON contains:
```json
{
  "frame_dir": "path/to/frames",
  "save_dir": "path/to/save",
  "store_dir": "path/to/cache",
  "store_mode": "load",  // or "store"
  "batch_size": 16,
  "clip_len": 750,
  "dataset": "soccernetballanticipation",
  
  // Architecture
  "feature_arch": "rny002",  // RegNet backbone
  "temporal_arch": "ed_sgp_mixer",
  "n_layers": 2,
  
  // Transformer
  "n_head": 8,
  "hidden_dim": 256,
  "n_encoder_layer": 6,
  "n_decoder_layer": 6,
  "n_query": 8,
  
  // Training
  "num_epochs": 100,
  "learning_rate": 0.001,
  "optimizer": "adamw",  // or "muon"
  "weight_decay": 0.05,
  
  // Task configuration
  "obs_perc": [0.2, 0.3, 0.5],  // Variable observation windows
  "pred_perc": 0.5,  // Anticipation window percentage
  "seg": true,  // Auxiliary segmentation task
  "anticipate": true,
  
  // Advanced options
  "actionness": false,  // Use actionness instead of EOS
  "use_anchors": false,  // Temporal anchor-based prediction
  "CALF_matching": false,  // Offset-based matching
}
```

---

## Main Training Pipeline

### Entry Point: `python main.py <config-path> <model-name> [--checkpoint-path <path>]`

**Flow:**
1. Load configuration from JSON
2. Initialize model (FUTR)
3. Load/create datasets (train, val)
4. Create data loaders
5. Setup optimizer (AdamW or Muon) + LR scheduler
6. Training loop:
   - Forward pass
   - Compute loss (action + offset + actionness)
   - Optional: CALF matching for prediction reordering
   - Backward pass
   - Validation every epoch
   - Save checkpoint if best mAP achieved
7. Final evaluation on test set
8. Log results to Weights & Biases

### Key Hyperparameters:
- **Warmup:** Linear warmup + Cosine annealing LR schedule
- **Loss Functions:** CrossEntropy, smoothing, or BCE
- **Class weighting:** Per-class weight support
- **Offset loss weight:** Temporal offset prediction scaling

---

## Utility Functions (`utils.py`)

### Loss Calculation:
- `cal_loss()` - CrossEntropy, label smoothing, or BCE
- `cal_performance()` - Accuracy and per-class statistics
- `cal_actionness_performance()` - Binary classification metrics

### Matching Algorithms:
- `CALF_matching()` - Hungarian algorithm for offset-based matching
- `CALF_matching2()` - Probability-based matching with actionness

### Helpers:
- `seed_everything()` - Reproducibility setup
- `normalize_offset()` - Temporal offset normalization

---

## Dataset Setup

### Download & Setup BAS Dataset:
```bash
python setup_dataset_BAS.py --download-key {NDA_KEY}
```

### Download & Setup BAA Dataset:
```bash
python setup_dataset_BAA.py --download-key {NDA_KEY}
```

**Requirements:** 
- NDA signature from https://www.soccer-net.org/data
- FFmpeg installed for frame extraction
- Storage: 100+ GB for full dataset with frames

---

## Evaluation

### Test on BAS Dataset:
```bash
python test.py config/SoccerNetBall/Base-Config-BAS.json \
                 model_checkpoint.pth \
                 model_name \
                 -s test
```

### Test on BAA Dataset:
```bash
python test.py config/SoccerNetBall/Base-Config-BAA.json \
                 model_checkpoint.pth \
                 model_name \
                 -s test  # or 'challenge'
```

### Evaluation Metrics:
- **mAP@δ:** Temporal localization precision
- **mAP@∞:** Action occurrence within window
- **Per-class F1:** Action-specific performance

---

## Citation

```bibtex
@InProceedings{Dalal_2025_CVPR,
    author    = {Dalal, Mohamad and Xarles, Artur and Cioppa, Anthony and Giancola, Silvio and Van Droogenbroeck, Marc and Ghanem, Bernard and Claupés, Albert and Escalera, Sergio and Moeslund, Thomas B.},
    title     = {Action Anticipation from SoccerNet Football Video Broadcasts},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {6079-6090}
}
```

---

## Key References

- **FUTR Paper:** Gong et al., "Future Transformer for Long-Term Action Anticipation" (CVPR 2022)
  - GitHub: https://github.com/gongda0e/FUTR
  
- **T-Deed Paper:** Xarles et al., "T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports Videos" (CVPR 2024)
  - GitHub: https://github.com/arturxe2/T-DEED_v2

- **SoccerNet:** Cioppa et al., Large-scale datasets for sports video understanding
  - Website: https://www.soccer-net.org/

---

## Project Status

- ✅ Main branch: Full implementation complete, tested
- ✅ Pose estimation: Experimental branch with two backends
- ⚠️ Instance segmentation: Work in progress
- 📊 Code tested on torch 2.8.0 (NVIDIA NGC container)

