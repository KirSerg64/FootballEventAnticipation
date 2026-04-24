# SAM 3 Model Weights

This directory holds the SAM 3 checkpoint used by
[`Sam3SegmentationTracker`](../../segmentation_tracking/sam3_wrapper.py).

## Download Instructions

SAM 3 weights are gated on Hugging Face and require approval from Meta.

1. **Request access** at: <https://huggingface.co/facebook/sam3>
2. **Authenticate** with the Hugging Face CLI:
   ```bash
   pip install huggingface_hub
   huggingface-cli login   # paste your access token
   ```
3. **Download** the checkpoint:
   ```bash
   python - <<'EOF'
   from huggingface_hub import hf_hub_download
   path = hf_hub_download(
       repo_id="facebook/sam3",
       filename="sam3.pt",
       local_dir="weights/sam3",
   )
   print(f"Saved to: {path}")
   EOF
   ```
4. Verify the file `weights/sam3/sam3.pt` exists.

## Package Installation

In addition to the weights, the `sam3` Python package must be installed:

```bash
git clone https://github.com/facebookresearch/sam3
cd sam3
pip install -e .
```

## Usage

```python
from segmentation_tracking import Sam3SegmentationTracker

tracker = Sam3SegmentationTracker(
    sam3_model_path="weights/sam3/sam3.pt",
    player_text_prompt="football player",
    ball_text_prompt="sports ball",
    field_text_prompt="football pitch",   # optional field segmentation
)
results = tracker.process_video("match.mp4")
```

Or via the CLI:

```bash
python scripts/run_segmentation_pose_tracking.py \
    --input match.mp4 \
    --output out.mp4 \
    --sam_backend sam3 \
    --sam3_model weights/sam3/sam3.pt \
    --sam3_player_prompt "football player" \
    --sam3_ball_prompt "sports ball" \
    --sam3_field_prompt "football pitch"
```
