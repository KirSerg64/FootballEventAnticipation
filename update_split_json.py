"""
update_split_json.py
---------------------
Updates the soccernetballanticipation split JSON files (train.json, val.json,
test.json) so that num_clips and num_frames reflect the clips currently listed
in the corresponding Labels-ball.json.

Intended to be run after setup_dataset_BAA_sample.py has pruned and filtered
the dataset to a smaller subset.

Usage example:
    python update_split_json.py \
        --data-path ./data/soccernetballanticipation \
        --frame-size 720p \
        --splits train,valid,test

The original JSON files are backed up as *_backup.json before being modified.
If a backup already exists it is not overwritten.
"""
import argparse
import json
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Each clip in the SoccerNet Ball Action Anticipation dataset contains exactly
# 750 frames (30 s at 25 fps).  Override with --frames-per-clip if needed.
DEFAULT_FRAMES_PER_CLIP = 750

# Mapping from logical split name → split JSON filename
SPLIT_JSON_MAP = {
    "train": "train.json",
    "valid": "val.json",
    "test":  "test.json",
}

# Mapping from logical split name → "video" field value stored in the JSON
SPLIT_VIDEO_NAME_MAP = {
    "train": "train",
    "valid": "valid",
    "test":  "test",
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def count_clips_from_labels(labels_path: Path) -> int:
    """
    Return the number of video entries in a Labels-ball.json file.

    Args:
        labels_path: Absolute path to Labels-ball.json.

    Returns:
        Number of entries in the 'videos' list.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If the JSON does not contain a 'videos' key.
    """
    with open(labels_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return len(data["videos"])


def update_split_json(
    split: str,
    data_path: Path,
    frame_size_path: str,
    frames_per_clip: int,
) -> None:
    """
    Update the split JSON for one split.

    Args:
        split: Split name ('train', 'valid', or 'test').
        data_path: Path to the soccernetballanticipation data directory.
        frame_size_path: Sub-directory name for the resolution ('224p' or '720p').
        frames_per_clip: Number of frames in each clip (used to compute num_frames).
    """
    # Locate Labels-ball.json
    labels_path = data_path / frame_size_path / split / "Labels-ball.json"
    if not labels_path.is_file():
        print(f"  [SKIP] Labels-ball.json not found at {labels_path}")
        return

    num_clips = count_clips_from_labels(labels_path)
    num_frames = num_clips * frames_per_clip

    # Locate the split JSON
    json_filename = SPLIT_JSON_MAP[split]
    json_path = data_path / json_filename
    if not json_path.is_file():
        print(f"  [SKIP] Split JSON not found at {json_path}")
        return

    # Read existing split JSON
    with open(json_path, "r", encoding="utf-8") as fh:
        split_data = json.load(fh)

    # split_data is a list with one entry, e.g.
    # [{"video": "train", "num_frames": 3440250, "num_clips": 4587}]
    if not isinstance(split_data, list) or len(split_data) == 0:
        print(f"  [SKIP] Unexpected format in {json_path}")
        return

    old_clips  = split_data[0].get("num_clips",  "?")
    old_frames = split_data[0].get("num_frames", "?")

    # Backup original (once)
    backup_path = json_path.with_stem(json_path.stem + "_backup")
    if not backup_path.is_file():
        shutil.copy2(json_path, backup_path)
        print(f"  Backed up {json_path.name} → {backup_path.name}")
    else:
        print(f"  Backup already exists ({backup_path.name}), skipping backup.")

    # Update fields
    split_data[0]["num_clips"]  = num_clips
    split_data[0]["num_frames"] = num_frames
    split_data[0]["video"]      = SPLIT_VIDEO_NAME_MAP[split]

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(split_data, fh, indent=2, ensure_ascii=False)

    print(
        f"  {json_path.name}: "
        f"num_clips {old_clips} → {num_clips}, "
        f"num_frames {old_frames} → {num_frames}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Update soccernetballanticipation split JSON files to match the clips\n"
            "currently present in Labels-ball.json (e.g. after sampling).\n"
            "Original files are backed up as *_backup.json."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/soccernetballanticipation",
        help="Path to the soccernetballanticipation directory (contains train.json etc.).",
    )
    parser.add_argument(
        "--frame-size",
        type=str,
        default="720p",
        choices=["224p", "448p", "720p"],
        help=(
            "Resolution folder to look for Labels-ball.json in.\n"
            "  224p → sub-folder '224p'\n"
            "  448p → sub-folder '720p'  (448p frames live in the 720p zip)\n"
            "  720p → sub-folder '720p'\n"
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="Comma-separated list of splits to process (default: train,valid,test).",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=DEFAULT_FRAMES_PER_CLIP,
        help=(
            f"Number of frames in each clip used to compute num_frames.\n"
            f"Default: {DEFAULT_FRAMES_PER_CLIP} (30 s × 25 fps)."
        ),
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # 448p frames are stored inside the 720p folder structure
    frame_size_path = "224p" if args.frame_size == "224p" else "720p"

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = set(splits) - set(SPLIT_JSON_MAP)
    if unknown:
        raise ValueError(
            f"Unknown split(s): {unknown}. Allowed: {set(SPLIT_JSON_MAP.keys())}"
        )

    print(f"Data path    : {data_path}")
    print(f"Resolution   : {args.frame_size} (folder: {frame_size_path})")
    print(f"Splits       : {splits}")
    print(f"Frames/clip  : {args.frames_per_clip}")
    print()

    for split in splits:
        print(f"--- {split} ---")
        update_split_json(split, data_path, frame_size_path, args.frames_per_clip)

    print("\nDone.")
