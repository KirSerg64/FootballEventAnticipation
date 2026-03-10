"""
visualize_clip_actions.py

Renders an annotated video for a selected clip, overlaying action labels
from Labels-ball.json at the corresponding timestamps.

Visual effects:
  - Colored border around the frame (bold at exact event frame, fading within window)
  - Semi-transparent top banner with action label + team (fades with distance)
  - Timeline strip at the bottom showing all event positions and current position

Usage examples:
  python visualize_clip_actions.py --clip clip_100
  python visualize_clip_actions.py --clip clip_1 --source video --fps 25 --window 20
  python visualize_clip_actions.py --clip clip_100 --output out.mp4 --types observation
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np


# ─── Colour palette ───────────────────────────────────────────────────────────
COLOURS = {
    "observation": (255, 120, 30),   # orange-blue (BGR)
    "anticipation": (60, 210, 60),   # green (BGR)
}

# Label colours for distinct simultaneous events
LABEL_PALETTE = [
    (0, 140, 255),   # orange
    (0, 220, 100),   # green
    (200, 50, 255),  # purple
    (0, 200, 200),   # cyan
    (255, 80, 80),   # blue-ish
    (255, 200, 0),   # light-blue
]

EXACT_BORDER_COLOUR  = (0, 50, 255)   # red (BGR) at exact event frame
FADE_BORDER_COLOUR   = (0, 140, 255)  # orange-yellow for nearby frames

TIMELINE_BG          = (30, 30, 30)
TIMELINE_HEIGHT      = 30
TIMELINE_CURSOR_COL  = (255, 255, 255)
FONT                 = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL           = cv2.FONT_HERSHEY_SIMPLEX


# ─── Helpers ──────────────────────────────────────────────────────────────────

def ms_to_frame(position_ms: int, fps: float) -> int:
    """Convert millisecond timestamp to 1-based frame index."""
    return max(1, round(position_ms / 1000.0 * fps))


def load_labels(labels_path: Path, clip_name: str, annotation_types: list[str]) -> list[dict]:
    """
    Parse Labels-ball.json and return a flat list of event dicts for the clip.
    Each dict: {frame, label, team, visibility, ann_type}
    """
    with open(labels_path, encoding="utf-8") as f:
        data = json.load(f)

    for video in data["videos"]:
        # path looks like "clip_100/720p.mp4"
        path_clip = video["path"].split("/")[0]
        if path_clip != clip_name:
            continue

        events = []
        for ann_type in annotation_types:
            entries = video["annotations"].get(ann_type, [])
            for entry in entries:
                events.append({
                    "position_ms": int(entry["position"]),
                    "label":       entry["label"],
                    "team":        entry.get("team", ""),
                    "visibility":  entry.get("visibility", ""),
                    "ann_type":    ann_type,
                })
        return events

    raise ValueError(f"Clip '{clip_name}' not found in {labels_path}")


def sort_frames(frame_dir: Path) -> list[Path]:
    """Return frame*.jpg files sorted by numeric index."""
    frames = list(frame_dir.glob("frame*.jpg"))
    frames.sort(key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    return frames


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_border(img: np.ndarray, alpha: float, colour: tuple) -> np.ndarray:
    """Draw a coloured border whose opacity scales with alpha [0..1]."""
    thickness = max(2, int(8 * alpha))
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), colour, thickness)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_banner(img: np.ndarray, events_at_frame: list[dict], alpha: float) -> np.ndarray:
    """Draw a semi-transparent top banner with action labels."""
    if not events_at_frame:
        return img

    h, w = img.shape[:2]
    banner_h = 42
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55 * alpha, img, 1 - 0.55 * alpha, 0)

    x = 10
    for i, ev in enumerate(events_at_frame):
        colour = COLOURS.get(ev["ann_type"], (200, 200, 200))
        tag = f"[{ev['ann_type'].upper()[:3]}]"
        text = f"{tag}  {ev['label']}  ({ev['team']})"

        # shadow
        cv2.putText(img, text, (x + 1, 29 + 1), FONT, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
        # main text
        cv2.putText(img, text, (x, 29), FONT, 0.62, colour, 2, cv2.LINE_AA)

        text_w, _ = cv2.getTextSize(text, FONT, 0.62, 2)[0]
        x += text_w + 30

    return img


def draw_timeline(img: np.ndarray, current_frame: int, total_frames: int,
                  events: list[dict], fps: float) -> np.ndarray:
    """Append a timeline strip below the image."""
    h, w = img.shape[:2]
    strip = np.full((TIMELINE_HEIGHT, w, 3), TIMELINE_BG, dtype=np.uint8)

    # Event tick marks
    for ev in events:
        ef = ms_to_frame(ev["position_ms"], fps)
        x  = int((ef - 1) / max(total_frames - 1, 1) * (w - 1))
        colour = COLOURS.get(ev["ann_type"], (200, 200, 200))
        cv2.line(strip, (x, 2), (x, TIMELINE_HEIGHT - 2), colour, 3)

    # Current position cursor
    cx = int((current_frame - 1) / max(total_frames - 1, 1) * (w - 1))
    cv2.line(strip, (cx, 0), (cx, TIMELINE_HEIGHT), TIMELINE_CURSOR_COL, 2)

    # Frame counter text
    label = f"frame {current_frame}/{total_frames}"
    cv2.putText(strip, label, (4, TIMELINE_HEIGHT - 6),
                FONT_SMALL, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    return np.vstack([img, strip])


def compute_active_events(events_with_frames: list[dict],
                          current_frame: int,
                          window: int) -> tuple[float, list[dict]]:
    """
    For the current frame, return (max_alpha, active_event_list).
    Alpha fades linearly from 1.0 (exact frame) to 0 at edge of window.
    """
    active = []
    max_alpha = 0.0

    for ev in events_with_frames:
        dist = abs(current_frame - ev["frame"])
        if dist <= window:
            alpha = 1.0 - dist / (window + 1)
            max_alpha = max(max_alpha, alpha)
            active.append({**ev, "_alpha": alpha})

    return max_alpha, active


# ─── Core rendering ───────────────────────────────────────────────────────────

def build_video_from_frames(frame_paths: list[Path], events_with_frames: list[dict],
                            total_frames: int, fps: float, window: int,
                            output_path: Path) -> None:
    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        sys.exit(f"Cannot read frame: {frame_paths[0]}")
    h, w = sample.shape[:2]
    out_h = h + TIMELINE_HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, out_h))

    for i, fp in enumerate(frame_paths):
        frame_no = i + 1
        img = cv2.imread(str(fp))
        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)

        max_alpha, active = compute_active_events(events_with_frames, frame_no, window)

        if max_alpha > 0:
            # Choose border colour: red on exact frame, orange nearby
            exact = [e for e in active if e["frame"] == frame_no]
            border_col = EXACT_BORDER_COLOUR if exact else FADE_BORDER_COLOUR
            img = draw_border(img, max_alpha, border_col)
            # Only show banner labels that are within half the window
            banner_events = [e for e in active if e["_alpha"] >= 0.5]
            img = draw_banner(img, banner_events, max_alpha)

        img = draw_timeline(img, frame_no, total_frames, events_with_frames, fps)
        writer.write(img)

        if frame_no % 50 == 0 or frame_no == total_frames:
            print(f"  {frame_no}/{total_frames} frames written", end="\r", flush=True)

    writer.release()


def build_video_from_video(video_path: Path, events_with_frames: list[dict],
                           fps: float, window: int, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_h = h + TIMELINE_HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, out_h))

    frame_no = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        frame_no += 1

        max_alpha, active = compute_active_events(events_with_frames, frame_no, window)

        if max_alpha > 0:
            exact = [e for e in active if e["frame"] == frame_no]
            border_col = EXACT_BORDER_COLOUR if exact else FADE_BORDER_COLOUR
            img = draw_border(img, max_alpha, border_col)
            banner_events = [e for e in active if e["_alpha"] >= 0.5]
            img = draw_banner(img, banner_events, max_alpha)

        img = draw_timeline(img, frame_no, total_frames, events_with_frames, fps)
        writer.write(img)

        if frame_no % 50 == 0:
            print(f"  {frame_no}/{total_frames} frames written", end="\r", flush=True)

    cap.release()
    writer.release()
    print(f"  {frame_no}/{total_frames} frames written")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise football action events overlaid on a clip.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--clip", required=True,
        help="Clip folder name, e.g. clip_100",
    )
    parser.add_argument(
        "--data_dir",
        default="data/soccernetballanticipation/720p/train",
        help="Base directory containing clip_* folders and Labels-ball.json "
             "(default: data/soccernetballanticipation/720p/train)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to Labels-ball.json (default: <data_dir>/Labels-ball.json)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output video file path (default: <clip>_annotated.mp4 in cwd)",
    )
    parser.add_argument(
        "--fps", type=float, default=25.0,
        help="Frames per second for timestamp→frame conversion and output video (default: 25)",
    )
    parser.add_argument(
        "--window", type=int, default=15,
        help="Number of frames before/after an event to show the overlay (default: 15)",
    )
    parser.add_argument(
        "--types", nargs="+",
        default=["observation", "anticipation"],
        choices=["observation", "anticipation"],
        help="Annotation types to include (default: observation anticipation)",
    )
    parser.add_argument(
        "--source", default="frames", choices=["frames", "video"],
        help="Use extracted JPEG frames ('frames') or the 720p.mp4 video ('video') as input "
             "(default: frames)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir   = Path(args.data_dir)
    clip_dir   = data_dir / args.clip
    labels_path = Path(args.labels) if args.labels else data_dir / "Labels-ball.json"
    output_path = Path(args.output) if args.output else Path(f"{args.clip}_annotated.mp4")

    # ── Validate paths ──
    if not clip_dir.is_dir():
        sys.exit(f"Clip directory not found: {clip_dir}")
    if not labels_path.is_file():
        sys.exit(f"Labels file not found: {labels_path}")

    # ── Load events ──
    print(f"Loading labels for '{args.clip}' from {labels_path} ...")
    events = load_labels(labels_path, args.clip, args.types)
    if not events:
        print("WARNING: No events found for this clip / annotation type combination.")

    # Attach frame numbers to events
    for ev in events:
        ev["frame"] = ms_to_frame(ev["position_ms"], args.fps)

    print(f"Found {len(events)} event(s):")
    for ev in events:
        print(f"  [{ev['ann_type']:>12}]  frame {ev['frame']:>5}  "
              f"({ev['position_ms']} ms)   {ev['label']}  [{ev['team']}]")

    # ── Render ──
    if args.source == "frames":
        frame_paths = sort_frames(clip_dir)
        if not frame_paths:
            sys.exit(f"No frame*.jpg files found in {clip_dir}")
        total_frames = len(frame_paths)
        print(f"\nRendering from {total_frames} JPEG frames → {output_path} ...")
        build_video_from_frames(frame_paths, events, total_frames, args.fps, args.window, output_path)

    else:  # video
        video_path = clip_dir / "720p.mp4"
        if not video_path.is_file():
            sys.exit(f"Video not found: {video_path}")
        print(f"\nRendering from video {video_path} → {output_path} ...")
        build_video_from_video(video_path, events, args.fps, args.window, output_path)

    print(f"\nDone!  Output saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
