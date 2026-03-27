# Proposal 4 Integration Guideline — Flow-Guided Temporal Attention Gating in FAANTRA

> **Status: design document — no code is implemented.**  
> This document is a step-by-step guide for integrating Proposal 4 from
> `integration_suggestions.md` into the FAANTRA codebase
> ([github.com/MohamadDalal/FAANTRA](https://github.com/MohamadDalal/FAANTRA)).  
> Implementation requires explicit approval before proceeding.

---

## Recap of the Idea

For every observation frame at position `t` in the input clip, compute a scalar
*temporal saliency score* `m[t]` from the SEA-RAFT optical flow field between
frames `t-1` and `t` (e.g. the spatial mean of the flow magnitude).  
Build a learned sigmoid gate from those scores and apply it element-wise to the
encoded frame tokens **before they enter the FUTR transformer encoder**:

```
m[t]  (scalar, per-frame flow magnitude summary)
  ↓
gate[t] = σ( W · normalize(m[t]) + b )   ∈ (0, 1)
  ↓
token[t] = token[t] * gate[t]             (broadcast over hidden_dim)
  ↓
FUTR transformer encoder
```

Frames with little motion (static set-pieces, camera cuts) are de-emphasised;
frames with rapid tactical shifts are amplified.

---

## Overview of Files to Touch

| # | File | Nature of change |
|---|---|---|
| A | `setup_dataset_BAA.py` / `setup_dataset_BAS.py` | Add offline flow-magnitude pre-computation pass |
| B | `dataset/frame.py` | Load and return flow magnitudes alongside frames |
| C | `model/flow_gate.py` *(new file)* | `FlowTemporalGate` nn.Module |
| D | `model/futr.py` | Instantiate gate; apply it in `forward()` |
| E | `opts.py` | Two new config keys |
| F | `configs/*.json` | Two new JSON entries |
| G | `train.py` / `train_dual.py` | Unpack & forward flow magnitudes |
| H | `eval.py` / `eval_BAA.py` | Same as G for inference paths |

---

## Detailed Change Descriptions

---

### A · Setup / Pre-processing Scripts (`setup_dataset_BAA.py`, `setup_dataset_BAS.py`)

**Goal:** produce a companion `flow_magnitudes.npy` file for every clip,
sitting next to the frame images, before training begins.

**What to add:**

1. After frames are extracted from each video clip (the existing `ffmpeg` step),
   add a second pass that iterates over consecutive frame pairs `(frame[t-1], frame[t])`.
2. For each pair, run SEA-RAFT inference (already available via
   `optical_flow/test_optical_flow.py`) to obtain a flow tensor
   `[2, H, W]` (u and v components).
3. Compute the per-frame scalar summary:
   ```
   magnitude = sqrt(u² + v²)       # shape [H, W]
   m[t]      = mean(magnitude)     # scalar  (or max — experiment both)
   ```
4. Collect all `m[t]` values for the clip into a 1-D NumPy array of shape
   `[num_frames,]`.
5. Save it as `flow_magnitudes.npy` inside the same clip directory where the
   frames live.

**Design notes:**

- The pre-computation can be done in a separate optional step (`--compute-flow`
  flag) so existing users are not forced to re-run setup.
- SEA-RAFT should run at reduced resolution (e.g. half or quarter of the
  training frame size) for speed — the magnitude summary does not need
  pixel-level precision.
- For the first frame of a clip (where `t-1` does not exist), set `m[0] = 0.0`
  or repeat `m[1]`.

---

### B · Dataset Class (`dataset/frame.py`)

**Goal:** load the pre-computed flow magnitudes and return them as an additional
field alongside the existing `(frames, labels)` output.

#### B1 · `ActionSpotDataset.__init__()`

Add a new optional parameter:
```
flow_dir : str | None = None
```
This is the root directory under which `flow_magnitudes.npy` files live, in the
same relative subdirectory structure as `frame_dir`.  
Store it as `self._flow_dir`.

#### B2 · `_store_clips_anticipation()` and `_store_clips()`

In addition to storing `self._frame_paths`, also build and store
`self._flow_paths` — a parallel list where each entry is the path to the
corresponding `flow_magnitudes.npy` file for that clip.  
If `self._flow_dir` is `None`, set each entry to `None` (graceful fallback for
runs without flow data).

#### B3 · `_load_clips()`

Load `flow_paths.pkl` from the store directory alongside `frame_paths.pkl`,
using the same pickling pattern.

#### B4 · `__getitem__()`

After loading the frame tensor for a given clip slice `[start:end]`, also:

1. Load `np.load(flow_path)` for the clip's flow magnitudes.
2. Slice it to the same `[start:end]` index range as the frames.
3. Convert to a `torch.FloatTensor` of shape `[S]` where `S` is the
   observation sequence length.
4. If `flow_path` is `None`, return a zero tensor `torch.zeros(S)` so the gate
   is identity (all-ones sigmoid output with `b = 0` init — see Section C).

The returned tuple becomes `(frames, labels, flow_magnitudes, ...)` — extend it
consistently with any existing extra fields returned.

#### B5 · `my_collate()`

Add `flow_magnitudes` to the collation, producing a batch tensor
`[B, S]` via `torch.stack`.

#### B6 · `get_datasets()` in `dataset/datasets.py`

Pass `flow_dir=args.flow_dir` through to `ActionSpotDataset(...)`.

---

### C · New Module: `model/flow_gate.py`

Create a new file containing a single `nn.Module`:

```
class FlowTemporalGate(nn.Module):

    Attributes
    ----------
    linear : nn.Linear(1 → 1)   — learnable scalar affine W, b
    init   : W = 1.0, b = 0.0   → gate ≈ 0.5 at start (neutral)

    forward(tokens, flow_magnitudes)
    --------------------------------
    tokens          : [B, S, D]   (encoded frame tokens after input_embed + relu)
    flow_magnitudes : [B, S]      (raw scalar per-frame summaries)
    returns         : [B, S, D]   (gated tokens)

    Steps inside forward():
    1. Normalise magnitudes across the sequence dimension:
       mu  = mean(flow_magnitudes, dim=1, keepdim=True)   [B, 1]
       std = std(flow_magnitudes,  dim=1, keepdim=True) + ε
       m_norm = (flow_magnitudes - mu) / std              [B, S]
    2. Expand for the linear layer:
       m_norm = m_norm.unsqueeze(-1)                      [B, S, 1]
    3. Compute gate:
       gate = sigmoid( linear(m_norm) )                   [B, S, 1]
    4. Apply gate:
       return tokens * gate                               [B, S, D]
```

**Design notes:**

- Using z-score normalisation per sequence instead of global normalisation
  makes the gate invariant to absolute magnitude levels (different match speeds,
  different clip lengths).  
- Initialising W=1, b=0 gives gate≈0.5 for a zero-mean input, which
  applies a uniform 50% suppression at the start of training. This is
  conservative — the model can quickly learn to differentiate.  
- An optional stronger initialisation: W=0, b=large positive → gate≈1.0
  (identity at the start), so the model has to actively learn to suppress.
  This is recommended for fine-tuning from an existing FAANTRA checkpoint.
- The module has only **2 learnable parameters** (W and b).

---

### D · FUTR Model (`model/futr.py`)

#### D1 · `__init__()`

Import the new module:
```python
from model.flow_gate import FlowTemporalGate
```

Add after the `self.input_embed` definition:
```python
if getattr(args, 'flow_gate', False):
    self.flow_gate = FlowTemporalGate()
else:
    self.flow_gate = None
```

#### D2 · `forward(inputs, mode='train')`

The `inputs` tuple currently is `(src, src_label)` in train mode or just `src`
in eval mode.  

Extend to `(src, src_label, flow_magnitudes)` / `(src, flow_magnitudes)`, or
— to be non-breaking — keep the existing tuple structure and pass
`flow_magnitudes` as a separate keyword argument:

```python
def forward(self, inputs, flow_magnitudes=None, mode='train'):
```

**Insertion point** — immediately after the existing lines:
```python
src = self.input_embed(src)   # [B, S, hidden_dim]
src = F.relu(src)
```

Add:
```python
if self.flow_gate is not None and flow_magnitudes is not None:
    src = self.flow_gate(src, flow_magnitudes)
```

Everything after this point (positional embeddings, rearrange, transformer call)
is unchanged.

---

### E · Configuration Arguments (`opts.py`)

In `update_args()`, add two lines:

```python
args.flow_gate = config.get('flow_gate', False)     # Enable/disable the gate
args.flow_dir  = config.get('flow_dir', None)       # Path to precomputed flow
```

---

### F · Config JSON Files (`configs/*.json`)

In whichever config is being used for a gated-flow run, add:

```json
"flow_gate": true,
"flow_dir": "/path/to/dataset/flow_magnitudes"
```

For baseline / ablation runs without flow, either omit these keys or set:
```json
"flow_gate": false
```

---

### G · Training Scripts (`train.py`, `train_dual.py`)

**Current unpacking line (in the training loop):**
```python
features, past_label, trans_off_future, trans_future_target, target_actionness = data
```

**Change to:**
```python
features, past_label, trans_off_future, trans_future_target, target_actionness, flow_magnitudes = data
flow_magnitudes = flow_magnitudes.to(device)
```

**Current model call:**
```python
outputs = model(inputs)
```

**Change to:**
```python
outputs = model(inputs, flow_magnitudes=flow_magnitudes)
```

The same pattern applies to `train_dual.py`.

---

### H · Evaluation Scripts (`eval.py`, `eval_BAA.py`)

Apply the same unpacking and model-call change as in Section G to all inference
loops in these scripts.

---

## Data Flow Summary

```
Video frames on disk
        │
        ▼
[SETUP SCRIPT - once]
SEA-RAFT(frame[t-1], frame[t]) → magnitude → m[t]
Save flow_magnitudes.npy  per clip
        │
        ▼
[DATALOADER - each batch]
ActionSpotDataset.__getitem__()
  → frames        [B, S, C, H, W]
  → flow_magnitudes [B, S]          ← new
        │
        ▼
[FUTR.forward()]
features = backbone(frames)         [B, S, input_dim]
src = input_embed(features) + relu  [B, S, hidden_dim]
src = FlowTemporalGate(src, m)      [B, S, hidden_dim]  ← new
src → rearrange → transformer encoder → decoder
  → action logits, offsets
```

---

## Compatibility with Existing Checkpoints

Loading a pre-trained FAANTRA checkpoint into the gated model is safe because:

1. `FlowTemporalGate` is an entirely new module — its weights are not present
   in the old checkpoint and will be initialised from scratch.
2. All other weights load cleanly since their keys are unchanged.
3. You should fine-tune for a few epochs (not full re-training) to let the
   gate calibrate while the backbone stays warm.

If loading a gated checkpoint back into the plain FUTR (for ablation), simply
set `flow_gate: false` in the config — the gate weights in the checkpoint will
be ignored via `strict=False` loading.

---

## Suggested Ablation Experiments

| Run | `flow_gate` | Flow available? | Purpose |
|---|---|---|---|
| Baseline | `false` | n/a | Current FAANTRA performance |
| Gate-all-zeros | `true` | zeros | Sanity check — should equal baseline |
| Gate-mean | `true` | spatial mean of magnitude | Main experiment |
| Gate-max | `true` | spatial max of magnitude | Alternative summary |
| Gate-ball-region | `true` | magnitude in ball bbox only | More targeted variant |
| Gate-frozen | `true` | mean, W=0 fixed | Effect of data vs. learned gate |

---

## Potential Pitfalls and Mitigations

| Risk | Mitigation |
|---|---|
| Very slow build-up phases (possession, set-pieces) have near-zero flow and get suppressed | Add a `min_gate` floor: `gate = min_gate + (1-min_gate) * sigmoid(...)` with `min_gate ≈ 0.1` |
| Camera pan dominates flow magnitude (not player motion) | Subtract global background flow before computing per-frame summary; or use the existing attractor / player-mask from `segmentation_tracking` to restrict the flow computation to the field region |
| Flow magnitude distribution varies significantly between match conditions (rain, pitch colour, broadcast style) | Per-sequence z-score normalisation (Section C, step 1) handles this automatically |
| Flow pre-computation is slow or large on disk | Compute at 1/4 resolution; magnitude summary is just one float per frame — the `flow_magnitudes.npy` file is negligible in size |
| Train/inference mismatch if flow is missing at inference | Return `flow_magnitudes=None` → `FlowTemporalGate` is bypassed → gate is identity → same output as baseline |

---

## Appendix: Which Flow Direction Should Be Paired with Frame t?

A key design choice that is not yet settled is whether the gate score `m[t]`
for frame `t` should be derived from:

- **Backward flow** — `flow(t-1 → t)`: motion that *arrived at* frame t, or  
- **Forward flow** — `flow(t → t+1)`: motion that *departs from* frame t.

This section analyses both options from first principles.

---

### Option A — Backward Flow `flow(t-1 → t)`

`m[t] = mean_magnitude( flow(t-1 → t) )`

**Semantic meaning:** "How much did the scene change *to produce* frame t?"  
The gate asks: *did this frame arrive after significant motion?*

**Advantages:**

1. **Strictly causal — no lookahead.**  
   When processing frame `t` you have already seen frame `t-1`, so
   `flow(t-1 → t)` is always available without peeking ahead.  
   This is the standard convention in offline pre-processing pipelines and
   eliminates any data-leakage concern at the boundary of the observation window.

2. **Straightforward boundary handling.**  
   The only edge case is the very first frame (`t = 0`), where there is no
   predecessor frame. Setting `m[0] = 0` (zero magnitude, gate ≈ neutral) is
   a natural and safe default.

3. **Standard in the optical-flow + video-understanding literature.**  
   OFF-ViNet (IEEE TIP 2024), which uses per-frame flow weights inside a
   video saliency transformer, uses backward flow for exactly this reason.

4. **"Landing" frame is still informative.**  
   A frame that arrives after a sprint, a tackle, or a fast counter-attack
   carry a lot of information precisely because they are high-motion outcomes.
   Amplifying such frames is sensible even for anticipation.

**Disadvantage:**

- The gate reflects *past* motion rather than *imminent* motion. A frame
  immediately before a shot build-up starts would receive a low gate value
  even though it is the most anticipation-relevant frame in the clip.

---

### Option B — Forward Flow `flow(t → t+1)`

`m[t] = mean_magnitude( flow(t → t+1) )`

**Semantic meaning:** "How much motion will *depart from* frame t?"  
The gate asks: *is this frame about to become a launch point for significant motion?*

**Advantages:**

1. **Directly anticipatory semantics.**  
   For an anticipation model the most critical frames are those just *before*
   an action starts. Forward flow for frame `t` is high precisely when motion
   is about to begin, so the gate amplifies the "trigger frames" rather than
   the "result frames."

2. **Better alignment with FAANTRA's goal.**  
   The FUTR decoder predicts what happens *after* the observation window.
   Forward-flow gating biases the encoder to store information about
   the dynamics that are about to unfold, giving the decoder a head start.

**Disadvantages:**

1. **Requires +1 frame lookahead at run time.**  
   To gate frame `t` with forward flow you need frame `t+1`.  
   For the *last* observation frame this means peeking one step into the
   anticipation window — a mild but real form of future leakage.  
   In offline pre-processing this is not a problem (all frames are on disk),
   but it complicates any online or streaming use-case.

2. **Last-frame boundary case is non-trivial.**  
   The final observation frame has no observation-window successor.
   The choices are: (a) use `m[T] = 0`, (b) use backward flow for the last
   frame only, or (c) allow the one-frame peek. Each option introduces
   an inconsistency.

---

### Option C — Centred / Symmetric Flow (compromise)

```
m[t] = 0.5 * mean_magnitude(flow(t-1 → t))
       + 0.5 * mean_magnitude(flow(t → t+1))
```

- Interior frames get a balanced view of incoming and outgoing motion.
- Edge frame `t = 0` falls back to forward flow only.
- Edge frame `t = T` falls back to backward flow only.
- One-frame lookahead is still required for interior frames; this option does
  not eliminate the practical concern of forward flow but smooths out the
  semantic asymmetry.

---

### Recommendation

**Primary choice: backward flow `flow(t-1 → t)`.**

Reasons:
1. Zero lookahead — the pre-computation loop can run strictly left-to-right
   over the frames already extracted by the setup scripts.
2. The only edge case (`m[0] = 0`) is trivial.
3. No boundary inconsistency at the observation-window cutoff.
4. The anticipation-relevant signal is not completely absent: a frame that
   arrived after high motion is also a frame from which the *next* high-motion
   segment is likely to continue (temporal autocorrelation of football dynamics).

**If offline pre-processing and a one-frame lookahead are acceptable:**  
Use forward flow `flow(t → t+1)` — this is the theoretically purer choice for
an anticipation task and is recommended as an ablation variant (see the
ablation table above).

**Suggested ablation entries to add:**

| Run | Flow direction | Purpose |
|---|---|---|
| Gate-backward | `flow(t-1 → t)` | Recommended default |
| Gate-forward | `flow(t → t+1)` | Anticipation-aligned alternative |
| Gate-centered | 0.5 × (backward + forward) | Symmetric compromise |

---

## References

- Proposal 4 concept: `optical_flow/integration_suggestions.md` (this repo)
- FAANTRA architecture: `model/futr.py` in [github.com/MohamadDalal/FAANTRA](https://github.com/MohamadDalal/FAANTRA)
- SEA-RAFT: [arxiv.org/abs/2405.14793](https://arxiv.org/abs/2405.14793)
- GFSalNet (gated fusion with flow, video saliency): [hucvl.github.io/GFSalNet](https://hucvl.github.io/GFSalNet/)
- SpikingVTG (saliency-feedback gating for video grounding): [openreview.net/pdf?id=30xMvMFtOA](https://openreview.net/pdf?id=30xMvMFtOA)
- OFF-ViNet (backward flow for frame weighting in video saliency transformer, IEEE TIP 2024): [ieeexplore.ieee.org/document/10508805](https://ieeexplore.ieee.org/document/10508805)
