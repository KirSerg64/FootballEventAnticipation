# Optical Flow Visualisation with SEA-RAFT

This directory contains a standalone test script that uses the
[SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) optical flow model
to process a video file and produce a **side-by-side visualisation video**
with the original frame on the left and the colour-coded optical flow on
the right.

---

## Quick-start

### 1. Clone SEA-RAFT

```bash
# From the repository root (or from this directory)
git clone https://github.com/princeton-vl/SEA-RAFT.git optical_flow/SEA-RAFT
```

> By default the script looks for the SEA-RAFT checkout at `SEA-RAFT/`
> **relative to your working directory**.  
> Override with `--sea_raft_dir /path/to/SEA-RAFT`.

### 2. Install dependencies

```bash
pip install -r optical_flow/requirements.txt
```

> SEA-RAFT itself is not a Python package; it is loaded by adding its
> `core/` directory to `sys.path` at runtime.

### 3. Run on a video

```bash
# Load model weights automatically from HuggingFace (recommended)
python optical_flow/test_optical_flow.py \
    --video  path/to/football_clip.mp4 \
    --output path/to/output_flow.mp4

# Use a locally downloaded checkpoint + config
python optical_flow/test_optical_flow.py \
    --video      path/to/football_clip.mp4 \
    --output     path/to/output_flow.mp4 \
    --model_path optical_flow/SEA-RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth \
    --config     optical_flow/SEA-RAFT/config/eval/spring-M.json
```

---

## All arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | *(required)* | Input video file path |
| `--output` | `<stem>_flow.mp4` | Output side-by-side video path |
| `--sea_raft_dir` | `SEA-RAFT` | Path to the cloned SEA-RAFT repository |
| `--model_url` | `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` | HuggingFace model ID |
| `--model_path` | `None` | Local `.pth` checkpoint (overrides `--model_url`) |
| `--config` | `None` | SEA-RAFT JSON config file (uses built-in spring-M defaults otherwise) |
| `--device` | auto | Inference device: `cpu` or `cuda` |
| `--iters` | 4 | Number of RAFT refinement iterations (higher = more accurate but slower) |
| `--scale` | -1 | Spatial scale exponent (`-1` = half resolution, `0` = full resolution) |
| `--frame_skip` | 1 | Process every N-th frame pair |
| `--max_frames` | 0 | Stop after this many output frames (0 = unlimited) |

---

## Output format

The output is a standard MP4 video with the same frame rate as the input.
Each frame is twice the width of the original:

```
┌──────────────────┬──────────────────┐
│                  │                  │
│  Original frame  │  Optical flow    │
│                  │  (colour wheel)  │
└──────────────────┴──────────────────┘
```

The optical flow is encoded using the standard Middlebury colour wheel:
motion direction maps to hue, speed maps to saturation.

---

## Downloading model weights manually

If you prefer not to use HuggingFace auto-download, weights can be fetched
from the SEA-RAFT
[Google Drive](https://drive.google.com/drive/folders/1YLovlvUW94vciWvTyLf-p3uWscbOQRWW)
or
[HuggingFace Model Hub](https://huggingface.co/papers/2405.14793).

Place the `.pth` file under `SEA-RAFT/models/` and pass it with
`--model_path`.

---

## References

```
@article{wang2024sea,
  title={SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow},
  author={Wang, Yihan and Lipson, Lahav and Deng, Jia},
  journal={arXiv preprint arXiv:2405.14793},
  year={2024}
}
```
