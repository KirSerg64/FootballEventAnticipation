"""
filter_labels_to_disk.py
------------------------
Filters Labels-ball.json for each split so that it only contains entries
whose clip folder actually exists on disk.

The original file is backed up as Labels-ball_backup.json (once; subsequent
runs skip the backup if it already exists).

Usage
-----
    # Default: all splits under data/soccernetballanticipation/720p/
    python filter_labels_to_disk.py

    # Specific resolution and/or splits:
    python filter_labels_to_disk.py --frame-size 720p --splits train,valid,test

    # Dry-run (print what would change without writing):
    python filter_labels_to_disk.py --dry-run
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def filter_split(split_dir: Path, dry_run: bool = False) -> None:
    """Filter Labels-ball.json in *split_dir* to entries with existing folders.

    Args:
        split_dir: Absolute path to the split directory (e.g. .../720p/train).
        dry_run:   If True, print the planned changes without writing anything.
    """
    labels_path = split_dir / "Labels-ball.json"
    if not labels_path.is_file():
        print(f"  [SKIP] Labels-ball.json not found in {split_dir}")
        return

    # Discover clip folders that actually exist on disk.
    existing_clips = {
        entry.name
        for entry in split_dir.iterdir()
        if entry.is_dir()
    }

    if not existing_clips:
        print(f"  [SKIP] No clip folders found in {split_dir}")
        return

    with open(labels_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    videos = data.get("videos", [])
    original_count = len(videos)

    # Keep an entry only when the first path component (the clip folder name)
    # matches a folder that exists on disk.
    # Entry paths look like "clip_42/720p.mp4".
    kept = [v for v in videos if Path(v["path"]).parts[0] in existing_clips]
    removed_count = original_count - len(kept)

    print(
        f"  {split_dir.name}: {original_count} entries → {len(kept)} kept "
        f"({removed_count} removed, {len(existing_clips)} folders on disk)"
    )

    if removed_count == 0:
        print("  Nothing to do.")
        return

    if dry_run:
        print("  [DRY RUN] No files written.")
        return

    # Backup original file (once).
    backup_path = labels_path.with_name("Labels-ball_backup.json")
    if not backup_path.is_file():
        shutil.copy2(labels_path, backup_path)
        print(f"  Backed up → {backup_path.name}")
    else:
        print(f"  Backup already exists ({backup_path.name}), skipping backup.")

    data["videos"] = kept
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(f"  Written {labels_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Filter Labels-ball.json files to only include entries whose clip "
            "folder exists on disk. Backs up originals as Labels-ball_backup.json."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/soccernetballanticipation",
        help="Root of the soccernetballanticipation data directory.",
    )
    parser.add_argument(
        "--frame-size",
        type=str,
        default="720p",
        choices=["224p", "448p", "720p"],
        help=(
            "Resolution sub-folder to look in.\n"
            "  224p → sub-folder '224p'\n"
            "  448p / 720p → sub-folder '720p'\n"
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test,challenge",
        help="Comma-separated list of splits to process (default: all four).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing any files.",
    )

    args = parser.parse_args()

    # 448p frames are stored in the 720p zip/folder structure.
    frame_size_folder = "224p" if args.frame_size == "224p" else "720p"
    root = Path(args.data_path) / frame_size_folder

    if not root.is_dir():
        raise FileNotFoundError(f"Resolution folder not found: {root}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    if args.dry_run:
        print("[DRY RUN MODE] No files will be written.\n")

    for split in splits:
        split_dir = root / split
        print(f"--- {split} ---")
        if not split_dir.is_dir():
            print(f"  [SKIP] Directory not found: {split_dir}")
            continue
        filter_split(split_dir, dry_run=args.dry_run)

    print("\nDone.")
