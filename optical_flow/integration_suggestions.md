# SEA-RAFT → FUTR Integration: Design Suggestions

> **Status: advisory document — no code changes yet.**  
> This document collects creative, research-backed proposals for feeding
> SEA-RAFT optical flow clues into the FUTR-based football action-anticipation
> pipeline.  
> None of these ideas are implemented here; implementation requires explicit
> approval.

---

## Context

| Component | Role |
|---|---|
| **SEA-RAFT** | Dense per-pixel optical flow between consecutive video frames — gives us a rich description of *who moves where and how fast* |
| **FUTR / FAANTRA** | Transformer encoder-decoder that reads a sequence of visual feature vectors extracted from past frames and predicts future action labels + durations |

The goal is to make the FUTR decoder pay more attention to the *most motion-rich / tactically significant* regions of each frame, and to give it an extra motion-aware feature channel that it can reason about over time.

---

## Proposal 1 — Flow-Magnitude Attention Bias (lightest-weight option)

**Core idea:** Convert the SEA-RAFT flow field for each frame pair into a scalar *saliency map* (flow magnitude, possibly Gaussian-smoothed), then inject it as a learned additive bias into FUTR's self-attention logits so that tokens corresponding to high-motion patches receive higher attention weight.

```
QK^T / sqrt(d)  +  alpha * flow_bias(t)   →   softmax   →   V
```

- `flow_bias(t)` is a `[T, H/p, W/p]` tensor (patch-level flow magnitude, linearly projected to match the token grid).
- `alpha` is a learnable scalar per head (or per layer) so the model can learn *how much* to trust motion.
- During the first epochs, `alpha` can be initialised to 0 so the bias kicks in gradually as the model learns its utility.

**Why it works:**  
Adding a content-agnostic positional-style bias to the attention logits is exactly how ALiBi (attention with linear biases) and relative-position encodings work. Here motion magnitude acts as a *temporal salience position* rather than spatial distance. Recent OFF-ViNet (IEEE TIP 2024) and GFSalNet validate the same principle in video saliency.

**Pros:** No new encoder needed; FUTR's backbone is untouched; differentiable end-to-end; negligible parameter count.  
**Cons:** Does not give FUTR explicit directional motion information, only magnitude.

---

## Proposal 2 — Flow Token Stream: Dual-Stream Encoder with Late Fusion

**Core idea:** Run two parallel FUTR-style encoders — one on the original RGB feature sequence (as today), one on a *flow feature sequence* extracted by a lightweight CNN or small ViT from SEA-RAFT colour-wheel images. Fuse the two encoded context vectors inside the FUTR decoder via a second cross-attention layer before predicting the next action.

```
RGB frames  ──► RGB encoder  ──► C_rgb
Flow fields ──► Flow encoder ──► C_flow
                                    │
FUTR decoder:   query × C_rgb  (first cross-attn)
                query × C_flow (second cross-attn, learnable gate α)
                     ↓
               action logits + durations
```

The gate `α` can be a per-head learned scalar or a small MLP (sigmoid-gated) that decides how much the decoder borrows from the flow context stream.

**Why it works:**  
This is the classical two-stream paradigm (Simonyan & Zisserman, NIPS 2014) updated with transformer fusion. FAANTRA already works well with RGB; the flow stream adds a complementary modality. InAViT (WACV 2024) and TAT-SARNet (MDPI 2024) both confirm dual-stream transformers outperform single-stream on anticipation tasks in sports.

**Pros:** Clean separation of RGB and motion reasoning; easy to ablate (zero out the flow stream); reuses the existing FUTR encoder architecture.  
**Cons:** ~2× encoder compute; precomputing and storing flow fields for the whole dataset requires significant disk space.

---

## Proposal 3 — Learnable Flow Encoder with Concatenated Feature Vector

**Core idea:** Build a dedicated *FlowEncoder* — a small 3D or 2+1D ConvNet (e.g. R(2+1)D-18) or a compact ViT operating on a stack of *N* consecutive flow fields — that produces a single fixed-size motion context vector `f ∈ R^d`. Concatenate this vector with FUTR's existing visual context vector before the decoder's cross-attention keys/values.

```
Video clip → Backbone (I3D / SlowFast) → v ∈ R^d
Flow stack → FlowEncoder              → f ∈ R^d
[v; f]  → linear projection → R^d  → FUTR decoder keys/values
```

This is the simplest, most drop-in-friendly option because it affects only the *input* to the decoder, leaving the decoder architecture fully intact.

**Why it works:**  
SEA-RAFT flow fields contain implicit information about player acceleration, direction changes (turns, sprints, sudden stops) that an RGB backbone may not capture reliably. Concatenating a flow context vector gives the decoder direct access to "how the game is physically developing" at training time. This mirrors how SlowFast networks fuse slow (appearance) and fast (motion) pathways.

**Pros:** Architecturally minimal; no attention modification required; can be pre-trained separately on flow-only self-supervised tasks and fine-tuned end-to-end.  
**Cons:** Temporal alignment between the RGB and flow contexts must be managed carefully (same clip boundaries).

---

## Proposal 4 — Flow-Guided Temporal Attention Gating (most targeted for anticipation)

**Core idea:** Compute a *temporal saliency score* for each frame in the input clip by taking the mean or max flow magnitude over the frame. Use these scores as soft gate weights applied to the encoded frame tokens *before* they enter FUTR's encoder.

```
flow_magnitude[t]  →  σ(W·m[t] + b)  =  gate[t]  ∈ (0,1)
token[t] = token[t] * gate[t]          (element-wise)
→ FUTR encoder (tokens suppressed on static frames)
```

This acts as a learned *frame-selection* mechanism: frames with little motion (camera cut or static set-piece) get de-emphasised, while frames with rapid tactical shifts (counter-attack, pressing, shot build-up) are amplified.

**Why it works:**  
In football, the anticipation-relevant moments are precisely those with high player and ball motion (breaks, presses, shooting runs). SpikingVTG (NeurIPS 2024) validates a saliency-feedback gating mechanism for video grounding. GFSalNet uses gated fusion of static and dynamic streams with impressive accuracy gains.

**Pros:** Very interpretable — the gate values become a visualisable, human-readable salience timeline; compatible with existing FUTR weights (the gate is inserted as a pre-processing layer).  
**Cons:** A single magnitude gate discards directional information; could suppress important slow tactical build-ups (patient possession before a through-ball).

---

## Proposal 5 — Flow-Warped Feature Propagation (OFF-ViNet style)

**Core idea:** Instead of running SEA-RAFT as a separate branch, use the flow field to *spatially warp* the previous frame's intermediate feature map before it is processed by the next temporal step of FUTR's encoder. This makes each token temporally consistent and already motion-compensated.

```
feat[t-1]  ──► warp(feat[t-1], flow[t-1→t])  ──┐
feat[t]    ────────────────────────────────────┤
                                               concat or add
                                                ↓
                                         encoder layer at time t
```

This is reminiscent of deformable convolutions and flow-based video prediction networks, and was used explicitly in OFF-ViNet (IEEE TIP 2024) to predict future saliency.

**Why it works:**  
In broadcast football footage the camera pans constantly. A flow-warped feature map removes apparent motion caused by the camera and preserves *ego-motion-corrected* player displacement, giving FUTR a cleaner signal about actual tactical movement.

**Pros:** Camera-motion correction as a side effect; reduces temporal jitter in feature space; enables the model to "look into the future" by warping the last observed features forward by one step.  
**Cons:** Requires flow estimation with sub-pixel accuracy (SEA-RAFT is well-suited); adds warping operations inside the encoder loop; gradients through the warp need `grid_sample` (differentiable but sometimes numerically unstable).

---

## Proposal 6 — Motion-Trajectory Cross-Attention Tokens (most football-specific, most novel)

**Core idea:** Extract *per-player motion trajectories* from SEA-RAFT (integrate flow vectors at tracked player keypoint locations over N frames) and encode each trajectory as a *trajectory token*. Inject these tokens as additional keys/values into FUTR's decoder cross-attention alongside the visual context tokens.

```
Visual context tokens:    C_1, C_2, … C_T        (from RGB encoder)
Trajectory tokens:        τ_p1, τ_p2, … τ_pK     (K players, from flow-tracked joints)
                               ↓
FUTR decoder cross-attention over [C_1..C_T, τ_p1..τ_pK]
                               ↓
Action prediction conditioned on both appearance and individual player dynamics
```

Each trajectory token `τ_pk` is formed by aggregating SEA-RAFT flow vectors sampled at the player's hip/shoulder keypoints (already available via the existing pose estimation pipeline) over a sliding window and projecting them through a small MLP.

**Why it works:**  
Football action anticipation is fundamentally about *individual and collective intent*. A striker sprinting into the box while the ball-carrier accelerates is a strong anticipation cue even before the pass is made. InAViT (WACV 2024) specifically shows that trajectory cross-attention — derived from optical flow — boosts egocentric action anticipation. The same principle transfers directly to broadcast football. The existing KeypointVelocityTracker (`segmentation_tracking/vector_field.py`) already integrates flow at keypoints using Farnebäck / CoTracker3 — upgrading that to SEA-RAFT flow is straightforward.

**Pros:** Richest signal; directly ties high-level tactical intent (player trajectories) to the action prediction; leverages existing keypoint and tracking infrastructure in the codebase; most differentiated from baseline FUTR.  
**Cons:** Requires reliable player tracking for all K players per frame; token count grows with team size; domain gap between the tracking quality and the training supervision may require careful curriculum learning.

---

## Proposal 7 — Flow-Conditioned Anticipation Query Initialisation

**Core idea:** In FUTR, the decoder uses learned anticipation *query embeddings* (one per future time step). Replace these static embeddings with *dynamic* ones computed as a function of the current flow state:

```
query[i]  =  static_embedding[i]  +  MLP(global_flow_summary)
```

where `global_flow_summary` is a pooled representation of the most recent SEA-RAFT flow field (e.g. mean and std of magnitude per spatial region, or a compact CNN encoding of the flow image).

**Why it works:**  
Static query embeddings force the decoder to rely entirely on cross-attention with the encoder for context. By conditioning queries on the current motion state, we give the model a "motion prior" that biases what kind of action it is about to predict — e.g., if global flow magnitude is very high (fast break), queries should be shifted toward *shot/cross/through-ball* rather than *set-piece/throw-in*.

**Pros:** Architecturally elegant; the motion context enters at the most impactful point (query initialisation governs what the decoder looks for); very few extra parameters.  
**Cons:** The `global_flow_summary` is a compression of a rich 2D field; may lose spatial detail unless a spatial-aware pooling (e.g., regional max-pooling over zone grids) is used.

---

## Comparison Table

| # | Strategy | Params added | Architectural invasiveness | Directional info | Football-specific | Difficulty |
|---|---|---|---|---|---|---|
| 1 | Flow-magnitude attention bias | ~1 scalar/head | Low | ✗ | ✗ | ★☆☆ |
| 2 | Dual-stream encoder + late fusion | ~1× encoder | Medium | ✓ | ✗ | ★★☆ |
| 3 | Learnable flow encoder + concat | ~1 small CNN | Low | ✓ | ✗ | ★☆☆ |
| 4 | Temporal gating on frame tokens | ~2 per frame | Low | ✗ | ✓ | ★☆☆ |
| 5 | Flow-warped feature propagation | 0 (warp op) | Medium | ✓ | ✓ | ★★★ |
| 6 | Per-player trajectory tokens | ~1 MLP/player | Medium | ✓ | ✓✓ | ★★★ |
| 7 | Flow-conditioned query init | ~1 small MLP | Low | partial | ✓ | ★★☆ |

---

## Recommended Starting Point

If I had to pick one approach to prototype first, I would suggest **Proposal 4 + Proposal 7 combined**:

1. Gate each encoded frame token by its temporal flow saliency score (Proposal 4) — this is a two-line change once flow is precomputed.
2. Condition the FUTR decoder queries on a compact global flow descriptor (Proposal 7) — this adds one small MLP and the existing SEA-RAFT pipeline already provides the flow.

Together these two minimal interventions address both *where to look in time* (gating) and *what to expect next* (query conditioning), touching the model at its two most sensitive points (encoder output and decoder input) with the smallest possible parameter budget.

Once that is validated, **Proposal 6 (per-player trajectory tokens)** is the most football-specific and likely highest-ceiling approach, building on the already-existing keypoint tracking infrastructure.

---

## References

- FUTR / FAANTRA: [arxiv.org/abs/2504.12021](https://arxiv.org/abs/2504.12021), [github.com/MohamadDalal/FAANTRA](https://github.com/MohamadDalal/FAANTRA)
- SEA-RAFT: [arxiv.org/abs/2405.14793](https://arxiv.org/abs/2405.14793)
- InAViT (trajectory cross-attention for egocentric anticipation, WACV 2024): [openaccess.thecvf.com](https://openaccess.thecvf.com/content/WACV2024/papers/Roy_Interaction_Region_Visual_Transformer_for_Egocentric_Action_Anticipation_WACV_2024_paper.pdf)
- OFF-ViNet (flow-warped feature saliency, IEEE TIP 2024): [ieeexplore.ieee.org/document/10508805](https://ieeexplore.ieee.org/document/10508805)
- GFSalNet (gated fusion for dynamic saliency): [hucvl.github.io/GFSalNet](https://hucvl.github.io/GFSalNet/)
- SpikingVTG (saliency-feedback gating for video grounding, NeurIPS 2024): [openreview.net/pdf?id=30xMvMFtOA](https://openreview.net/pdf?id=30xMvMFtOA)
- TAT-SARNet (dual-stream transformer for soccer, MDPI 2024): [mdpi.com/2227-7390/13/18/3011](https://www.mdpi.com/2227-7390/13/18/3011)
- FootBots / TranSPORTmer (football trajectory transformers, 2024): [arxiv.org/abs/2406.19852](https://arxiv.org/html/2406.19852v1)
- ALiBi (additive attention biases): [arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409)
