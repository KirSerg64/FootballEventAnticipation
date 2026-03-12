# Identification Quality Improvement Plan

> **Status: IMPROVEMENTS 1 + 2 + 3 + 4 + 6 + 7 IMPLEMENTED.**
>
> Improvements 5 (Appearance Re-ID / OSNet) and 8 (custom standalone Kalman
> tracker) were not approved and remain unimplemented.
>
> See `segmentation_tracking/ball_kalman.py`, `segmentation_tracking/team_classifier.py`,
> and the updated `segmentation_tracking/segmentation_model.py`,
> `segmentation_tracking/association.py`, `segmentation_tracking/visualization.py`,
> and `scripts/run_segmentation_pose_tracking.py` for the changes.

---

## 1. Root-Cause Analysis

### 1.1 SAM2 is seeded once on frame 0 only

`SegmentationTracker.process_video()` calls YOLO once on the very first frame,
feeds those bounding boxes to `SAM2VideoPredictor`, and then never re-initialises
the predictor with new prompts.

**Consequences:**
- Players who are partially visible, blurry, or misdetected on frame 0 receive a
  poor initial mask that drifts throughout the video.
- Players entering the scene after frame 0 are handled only by the lightweight
  `_redetect_new_players` path (rectangle masks, no SAM2 tracking).
- Camera cuts / large zooms break SAM2's memory propagation completely because the
  visual context changes abruptly.

### 1.2 No appearance (Re-ID) model

Player IDs are assigned strictly by detection order on frame 0, then propagated
by SAM2's learned memory.  There is no appearance embedding (jersey colour, body
silhouette, texture) stored per track.

**Consequences:**
- After any occlusion or mask failure, when a player "reappears" they are given a
  brand-new ID (via `_next_player_id`) instead of recovering the original one.
- Two players who swap positions temporarily will exchange IDs with no mechanism
  to correct the mistake.

### 1.3 Greedy IoU matching (not globally optimal)

`_assign_new_player_ids` iterates over detections in arbitrary order and greedily
takes the first match exceeding `iou_threshold`.

**Consequences:**
- In dense football scenes with 20+ players, bounding boxes heavily overlap.  A
  greedy algorithm systematically makes sub-optimal assignments.  The Hungarian
  (linear-sum-assignment) algorithm finds the globally optimal one-to-one pairing
  at negligible extra cost.
- Fixed `iou_threshold = 0.5` is too strict for tightly packed crowds (players
  standing shoulder-to-shoulder rarely achieve IoU > 0.5 between detection and
  predicted position).

### 1.4 No Kalman-filter motion prediction

The pipeline performs no temporal motion modelling.  Between frames, each player's
position is treated as unknown.

**Consequences:**
- At 25–50 fps a player can move 20–60 pixels per frame.  Without a velocity
  estimate the IoU between the track's last known bbox and the new detection can
  drop to zero even with no occlusion.
- During occlusions (which are constant in football) the track has no predicted
  location to hold it alive.

### 1.5 No track-lifecycle management

The code has no concept of *tentative* vs *confirmed* tracks, no maximum *track
age* before deletion, and no minimum *hit count* before a new ID is emitted.

**Consequences:**
- Spurious detections (referee, advertising hoardings, misdetections) immediately
  get a permanent player ID.
- Tracks lost for more than one frame are never coalesced back with their previous
  IDs, causing an ever-growing ID counter.

### 1.6 Camera-motion not compensated

Fast panning, zooming, and broadcast cuts shift all player positions simultaneously.
The IoU between frame *t* and frame *t+1* bboxes can be zero even though the same
players are fully visible in both.

**Consequences:**
- After any camera movement, IoU-based matching fails completely and every player
  gets a new ID.

### 1.7 Fallback masks are rectangles

When SAM2VideoPredictor fails or is unavailable, `_fallback_detect` fills the
entire bounding box with a solid rectangle as the segmentation mask.

**Consequences:**
- Rectangular masks overlap heavily with neighbours, making the subsequent
  IoU-based pose-to-mask association in `association.py` unreliable.

### 1.8 Ball tracking is stateless

Ball detection is an independent YOLO call every frame with no Kalman filter, no
confirmed track, and no recovery after a missed detection.

**Consequences:**
- Fast-moving balls cause missed detections; the ball position jumps around or
  disappears for several frames.

---

## 2. Proposed Improvements

Improvements are ordered from highest to lowest expected impact.

---

### Improvement A — Replace greedy matching with Hungarian algorithm  *(High impact, low effort)*

**Where:** `segmentation_tracking/association.py` and
`segmentation_tracking/segmentation_model.py` (`_assign_new_player_ids`).

**What to change:**
Replace the nested-loop greedy matching in `_assign_new_player_ids` and
`associate_poses_with_tracks` with `scipy.optimize.linear_sum_assignment`.
Build a cost matrix of shape `(n_tracks × n_detections)` populated with
`1 − IoU(track_i, det_j)`.  The Hungarian solver returns the globally optimal
one-to-one assignment in O(n³) time, which is fast for n ≤ 30 players.

**Expected result:**
- Eliminates cascading wrong assignments in dense scenes.
- Requires only `scipy` (already in `requirements.txt`).

---

### Improvement B — Add appearance Re-ID embeddings  *(High impact, medium effort)*

**Where:** new file `segmentation_tracking/reid_model.py`, called from
`segmentation_tracking/segmentation_model.py`.

**What to change:**

1. After YOLO detects players (frame 0 and re-detection frames), crop each
   player's bounding-box region from the frame.
2. Pass the crops through a lightweight Re-ID network (e.g., **OSNet-0.25** from
   the `torchreid` library, or the built-in appearance extractor shipped with
   **BoT-SORT** in ultralytics).
3. Store a rolling gallery of appearance embeddings per track ID (e.g., an
   exponential moving average of the last 30 crop embeddings).
4. When a detection cannot be matched by IoU, match it against the gallery using
   cosine similarity.  If similarity > threshold, reuse the historical track ID
   instead of issuing a new one.

**Expected result:**
- Players who were occluded and re-appear are recognised by their jersey and
   assigned the correct historical ID.
- Particularly powerful after camera cuts.

**Models to consider:**
| Model | Size | Speed (GPU) |
|-------|------|------------|
| OSNet-0.25 (`torchreid`) | 0.6 MB | ~1 ms/crop |
| BoT-SORT appearance head | built into ultralytics | ~2 ms/crop |
| FastReID (ResNet-50) | 94 MB | ~5 ms/crop |

OSNet-0.25 is recommended as a first step due to minimal footprint.

---

### Improvement C — Kalman filter motion prediction  *(High impact, medium effort)*

**Where:** new file `segmentation_tracking/kalman_tracker.py`, integrated into
`segmentation_tracking/segmentation_model.py`.

**What to change:**

Implement a per-track 8-dimensional Kalman filter
`(cx, cy, aspect_ratio, height, vcx, vcy, va, vh)` — the same state model used
by SORT / DeepSORT / ByteTrack.

For each frame:
1. **Predict** each active track's bbox for the current frame using the filter.
2. **Match** predicted bboxes against new YOLO detections (Hungarian + IoU,
   with the predicted bbox as the "prior").
3. **Update** matched tracks with the observed detection.
4. **Mark unmatched** tracks as tentative; keep them alive for up to
   `max_age` frames using the prediction alone.

**Expected result:**
- Tracks survive short occlusions (1–5 frames) without ID change.
- Camera panning is partially absorbed because the filter velocity adapts.
- Players moving quickly across the frame are still matched correctly.

**Note:** `ultralytics` already ships **BoT-SORT** and **ByteTrack** trackers that
can be enabled with a single YAML config.  Using them directly would provide
Kalman filtering, Hungarian matching, Re-ID embedding support, and track lifecycle
management all at once (see Improvement E).

---

### Improvement D — Camera-motion compensation (homography)  *(High impact, medium effort)*

**Where:** `segmentation_tracking/segmentation_model.py`.

**What to change:**

Before computing IoU between consecutive frames:
1. Detect sparse keypoints (ORB or SIFT) on the pitch/background region (not on
   players).
2. Estimate the frame-to-frame homography `H` using RANSAC.
3. Warp all previous-frame bounding boxes through `H` to get "motion-corrected"
   predicted positions.
4. Use these corrected positions as the prior for IoU matching instead of the
   raw last-known boxes.

OpenCV provides `cv2.findHomography` and `cv2.perspectiveTransform` which makes
this straightforward to add.

**Expected result:**
- IoU matching quality is preserved during camera panning and zooming.
- ID switches caused purely by camera movement are eliminated.

---

### Improvement E — Switch primary tracker to BoT-SORT / ByteTrack  *(High impact, low additional effort after A–D)*

**Where:** `segmentation_tracking/segmentation_model.py`.

**What to change:**

`ultralytics` exposes a unified tracking API:

```python
results = yolo_model.track(
    source=video_path,
    tracker="botsort.yaml",   # or "bytetrack.yaml"
    persist=True,
    conf=0.25,
    classes=[0],              # person only
)
```

Each result has `boxes.id` — the persistent track ID maintained by BoT-SORT's
Kalman filter + Hungarian matching + optional Re-ID.

**Proposed pipeline restructure:**

```
Frame N
  ├── BoT-SORT track() → person bboxes with stable IDs (Kalman + Hungarian + Re-ID)
  │       ↓
  ├── SAM2VideoPredictor → refine detections to pixel-level masks
  │   (seeded with BoT-SORT bboxes instead of raw YOLO bboxes)
  │       ↓
  ├── YOLO pose → keypoints
  │       ↓
  └── Association (mask ↔ keypoints, BoT-SORT IDs are ground truth)
```

**Why this is better:**
- BoT-SORT already implements improvements A + C + appearance Re-ID.
- SAM2 is used only for per-frame mask quality, not for tracking logic; it becomes
  a "segmentation refiner" rather than the source of truth for IDs.
- The overall architecture is cleaner and each component does what it does best.

---

### Improvement F — Team color clustering  *(Medium impact, low effort)*

**Where:** new helper function in `segmentation_tracking/association.py` or a new
`segmentation_tracking/team_classifier.py`.

**What to change:**

1. For each confirmed player track, extract the torso region (between shoulder and
   hip keypoints, or upper half of the mask bbox).
2. Compute the dominant HSV colour via K-means (k=3: team A, team B, referee).
3. Store team label per track ID.

**Expected results:**
- Players from the same team are more likely to be distinguishable from the
  opposite team, reducing inter-team ID confusion.
- The team label can be surfaced in the visualization and JSON export.
- When Re-ID embeddings are ambiguous, team colour provides an additional signal.

---

### Improvement G — Track lifecycle management  *(Medium impact, low effort)*

**Where:** `segmentation_tracking/segmentation_model.py`.

**What to change:**

Introduce three track states:
- **Tentative**: detected for fewer than `min_hits` consecutive frames.
  Do not emit an ID yet.
- **Confirmed**: `hit_streak >= min_hits`.  Emits a stable ID.
- **Lost**: not matched for `max_age` frames.  ID retired; will not be reused.

Suggested defaults: `min_hits = 3`, `max_age = 30` (1 second at 30 fps).

**Expected results:**
- Spurious detections no longer pollute the ID space.
- Truly lost players are not confused with re-entering ones.

---

### Improvement H — Improved ball tracking with Kalman filter  *(Medium impact, low effort)*

**Where:** `segmentation_tracking/segmentation_model.py`.

**What to change:**

Apply a 4-state Kalman filter `(cx, cy, vcx, vcy)` to the ball centre detections:
- Predict ball position every frame.
- Update only when a YOLO detection is available.
- Report the filtered position (not the raw detection) as `ball_center`.

Use SAM2 point-prompt segmentation with the predicted ball centre as input for the
ball mask instead of the current ellipse approximation.

---

### Improvement I — Domain-specific YOLO model  *(Lower impact, higher setup cost)*

**Where:** `segmentation_tracking/segmentation_model.py` (model path).

**What to change:**

Replace `yolo11x.pt` (COCO-pretrained) with a YOLO model fine-tuned on football
broadcast footage.  Suitable starting points:

- **SoccerNet Detection** annotations (freely available).
- **Roboflow Football Player Detection** dataset (community-contributed).
- Fine-tune for 50–100 epochs on player + ball + referee classes.

**Expected result:**
- Better recall for partially occluded players, small-scale distant players, and
  players near the edge of the pitch.

---

## 3. Recommended Implementation Order

| Priority | Improvement | Files affected | Status |
|----------|------------|----------------|--------|
| 1 | **E** – BoT-SORT / ByteTrack as primary tracker | `segmentation_model.py` | ✅ Implemented |
| 2 | **A** – Hungarian matching | `association.py`, `segmentation_model.py` | ✅ Implemented |
| 3 | **G** – Track lifecycle management | `segmentation_model.py` | ✅ Implemented |
| 4 | **D** – Camera-motion compensation | `segmentation_model.py` | ✅ Implemented |
| 5 | **B** – Appearance Re-ID (OSNet) | new `reid_model.py` | ⏸ Not approved |
| 6 | **H** – Ball Kalman filter | `ball_kalman.py`, `segmentation_model.py` | ✅ Implemented |
| 7 | **F** – Team color clustering | `team_classifier.py` | ✅ Implemented |
| 8 | **C** – Custom Kalman filter (if not using BoT-SORT) | new `kalman_tracker.py` | ⏸ Not approved |
| 9 | **I** – Domain-specific YOLO | model weights only | ⏸ Not approved |

---

## 4. New CLI Parameters (implemented)

The following parameters were added to
`scripts/run_segmentation_pose_tracking.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tracker` | `botsort` | Tracking algorithm: `botsort` or `bytetrack` |
| `--max_age` | `30` | Frames a track survives without a detection |
| `--no_homography` | off | Disable camera-motion compensation |
| `--team_colors` | off | Enable team-color clustering |
| `--n_teams` | `2` | Number of K-means team clusters |
| `--team_refit_interval` | `30` | Re-run K-means every N frames |

---

## 5. What Is NOT Being Changed

- The overall pipeline structure (segmentation → pose → association → visualization)
  remains the same.
- The `PlayerTrack`, `BallTrack`, and `SegmentationResult` data structures are
  backwards-compatible.
- The JSON export format is unchanged.
- The `visualization.py` module requires no changes (it is independent of the
  tracking algorithm).
- No existing tests are removed.

---

*Please review and approve this plan.  Once approved, implementation will proceed
in the priority order listed in Section 3, with a `report_progress` commit after
each completed improvement.*
