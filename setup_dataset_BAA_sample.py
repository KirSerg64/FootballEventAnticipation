"""
setup_dataset_BAA_sample.py
----------------------------
Variant of setup_dataset_BAA.py that supports randomly sampling a fixed number
of clips per split.  Non-sampled clips are deleted from disk.  Labels-ball.json
is backed up and then filtered to only contain the sampled clips.

Usage example:
    python setup_dataset_BAA_sample.py \
        --export-only \
        --download-path ./data/soccernetballanticipation \
        --frame-size 720p \
        --sample train:200,valid:50,test:50 \
        --seed 42
"""
import argparse
import json
import os
import pyzipper
import random
import shutil
import subprocess
import multiprocessing as mp
from itertools import repeat
from pathlib import Path
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------------------
# Download / extract helpers (unchanged from setup_dataset_BAA.py)
# ---------------------------------------------------------------------------

def download_split(split, download_path, frame_size="224p"):
    """
    Download the specified split of the dataset.

    Args:
        split (str): The split to download (train, valid, test, or challenge).
        download_path (str): The directory where the dataset will be downloaded.
        frame_size (str): Resolution folder to download (224p or 720p).

    Returns:
        None
    """
    print(f"Downloading {split} split")
    split_path = Path(os.path.join(download_path, frame_size, split + ".zip"))
    if split_path.is_file():
        print(f"Split {split} already downloaded. Skipping download.")
    else:
        snapshot_download(
            repo_id="SoccerNet/ActionAnticipation",
            repo_type="dataset",
            revision="main",
            local_dir=download_path,
            allow_patterns=[f"{frame_size}/*" + split + ".zip"],
        )


def extract_split(split, download_key, download_path, delete_videos=False):
    """
    Extract the specified split of the dataset.

    Args:
        split (str): The split to extract (train, valid, test, or challenge).
        download_key (str): AES password for the zip archive.
        download_path (str): The directory where the dataset is downloaded.
        delete_videos (bool): Whether to delete the zip file after extraction.

    Returns:
        None
    """
    print(f"Extracting {split} split")
    split_path = Path(os.path.join(download_path, split + ".zip"))
    with pyzipper.AESZipFile(split_path, "r") as zf:
        zf.extractall(split_path.parent, pwd=download_key.encode())
    if delete_videos:
        os.remove(split_path)
        print(f"Deleted {split} zip file to save space")


# ---------------------------------------------------------------------------
# Frame-export helpers
# ---------------------------------------------------------------------------

def export_clip(clip, split, delete_videos, low_res, download_path, frame_size, use_cuda=True):
    """
    Export frames from a single clip using the specified resolution.

    Args:
        clip (str): The folder name of the clip to export frames from.
        split (str): The split name (train, valid, test, …).
        delete_videos (bool): Whether to delete the source video after exporting.
        low_res (bool): True when the input video is 224p.
        download_path (str): Root directory of the resolution folder (e.g. .../720p).
        frame_size (str): Target export resolution label (224p, 448p, or 720p).
        use_cuda (bool): Use CUDA hardware-accelerated decoding via ffmpeg.

    Returns:
        None
    """
    resolution = "224p.mp4" if low_res else "720p.mp4"
    video_path = Path(os.path.join(download_path, split, clip, resolution))
    print(f"Exporting {video_path}")

    if use_cuda:
        if frame_size == "448p":
            subprocess.call([
                "ffmpeg", "-hwaccel", "cuda", "-i", str(video_path),
                "-q:v", "1", "-vf", "scale=796x448",
                os.path.join(str(video_path.parent), "frame%d.jpg"),
            ])
        else:
            subprocess.call([
                "ffmpeg", "-hwaccel", "cuda", "-i", str(video_path),
                "-q:v", "1", "-vf", "scale=398x224",
                os.path.join(str(video_path.parent), "frame%d.jpg"),
            ])
    else:
        if frame_size == "448p":
            subprocess.call([
                "ffmpeg", "-i", str(video_path),
                "-q:v", "1", "-vf", "scale=796x448",
                os.path.join(str(video_path.parent), "frame%d.jpg"),
            ])
        else:
            subprocess.call([
                "ffmpeg", "-i", str(video_path),
                "-q:v", "1", "-vf", "scale=398x224",
                os.path.join(str(video_path.parent), "frame%d.jpg"),
            ])

    if delete_videos and video_path.is_file():
        os.remove(video_path)
        print(f"Deleted {video_path} to save space")


def export_frames(split, download_path, clips, delete_videos=False, frame_size="448p", num_cpus=4, use_cuda=True):
    """
    Export frames from the specified clips in parallel.

    Args:
        split (str): The split name.
        download_path (str): Root directory of the resolution folder.
        clips (list[str]): Clip folder names to export.
        delete_videos (bool): Delete source videos after export.
        frame_size (str): Target export resolution label.
        num_cpus (int): Number of parallel workers.
        use_cuda (bool): Use CUDA-accelerated ffmpeg decoding.

    Returns:
        None
    """
    print(f"Exporting frames from {split} split ({len(clips)} clips)")
    if not clips:
        print(f"  No clips to export in {split}, skipping.")
        return
    low_res = frame_size == "224p"
    with mp.Pool(num_cpus) as p:
        p.starmap(
            export_clip,
            zip(
                clips,
                repeat(split),
                repeat(delete_videos),
                repeat(low_res),
                repeat(download_path),
                repeat(frame_size),
                repeat(use_cuda),
            ),
        )


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def get_clips_in_split(split_dir: str) -> list[str]:
    """Return sorted list of clip folder names inside *split_dir*."""
    try:
        return sorted(next(os.walk(split_dir))[1])
    except StopIteration:
        return []


def sample_and_prune_clips(split, download_path, n_samples, seed=42):
    """
    Randomly sample *n_samples* clip folders from a split and delete the rest.

    Args:
        split (str): Split name (train, valid, test, …).
        download_path (str): Root directory of the resolution folder.
        n_samples (int | None): How many clips to keep; None means keep all.
        seed (int): Random seed for reproducibility.

    Returns:
        list[str]: Sorted list of sampled clip folder names.
    """
    split_dir = os.path.join(download_path, split)
    all_clips = get_clips_in_split(split_dir)

    if not all_clips:
        print(f"  No clip folders found in {split_dir}. Nothing to sample.")
        return []

    if n_samples is None or n_samples >= len(all_clips):
        print(f"  Keeping all {len(all_clips)} clips in '{split}' split (no sampling).")
        return all_clips

    rng = random.Random(seed)
    sampled = sorted(rng.sample(all_clips, n_samples))
    to_delete = [c for c in all_clips if c not in set(sampled)]

    print(
        f"  '{split}': sampled {len(sampled)}/{len(all_clips)} clips. "
        f"Deleting {len(to_delete)} unsampled clip folders…"
    )
    for clip in to_delete:
        clip_path = os.path.join(split_dir, clip)
        shutil.rmtree(clip_path, ignore_errors=True)

    return sampled


def filter_labels_json(split, download_path, sampled_clips):
    """
    Filter Labels-ball.json to only include entries for *sampled_clips*.
    The original file is backed up as Labels-ball_backup.json (once).

    Args:
        split (str): Split name, used to locate the JSON file.
        download_path (str): Root directory of the resolution folder.
        sampled_clips (list[str]): Clip folder names to keep.

    Returns:
        None
    """
    labels_path = Path(os.path.join(download_path, split, "Labels-ball.json"))
    if not labels_path.is_file():
        print(f"  Labels-ball.json not found at {labels_path}. Skipping label filtering.")
        return

    # Backup original (only once — skip if backup already exists)
    backup_path = labels_path.with_name("Labels-ball_backup.json")
    if not backup_path.is_file():
        shutil.copy2(labels_path, backup_path)
        print(f"  Backed up Labels-ball.json → {backup_path.name}")
    else:
        print(f"  Backup already exists ({backup_path.name}), skipping backup.")

    with open(labels_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    sampled_set = set(sampled_clips)
    original_count = len(data.get("videos", []))

    # Each entry has "path" like "clip_1/720p.mp4"; the first path component
    # is the clip folder name.
    filtered = [
        v for v in data.get("videos", [])
        if Path(v["path"]).parts[0] in sampled_set
    ]
    data["videos"] = filtered

    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(
        f"  Labels-ball.json updated: {original_count} → {len(filtered)} entries "
        f"({original_count - len(filtered)} removed)."
    )


# ---------------------------------------------------------------------------
# Argument parsing helper
# ---------------------------------------------------------------------------

def parse_sample_arg(sample_str):
    """
    Parse a string of the form "train:100,valid:30,test:50" into a dict.

    Returns:
        dict[str, int]: Mapping from split name to number of clips to sample.

    Raises:
        argparse.ArgumentTypeError: On malformed input.
    """
    result = {}
    if not sample_str:
        return result
    for part in sample_str.split(","):
        part = part.strip()
        if ":" not in part:
            raise argparse.ArgumentTypeError(
                f"Invalid --sample token '{part}'. Expected format: 'split:N' (e.g. train:100)."
            )
        split_name, raw_n = part.split(":", 1)
        split_name = split_name.strip()
        try:
            n = int(raw_n.strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid clip count '{raw_n}' for split '{split_name}'. Must be an integer."
            )
        if n <= 0:
            raise argparse.ArgumentTypeError(
                f"Clip count for split '{split_name}' must be a positive integer, got {n}."
            )
        result[split_name] = n
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download, extract, and/or export a (sampled) subset of the SoccerNet "
            "Ball Action Anticipation dataset.\n\n"
            "Required storage (approx., with --delete-videos):\n"
            "  720p : train 323 GB | valid 85 GB | test 164 GB | challenge 153 GB\n"
            "  448p : train 175 GB | valid 46 GB | test  89 GB | challenge  82 GB\n"
            "  224p : train  57 GB | valid 15 GB | test  29 GB | challenge  27 GB\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--download-key",
        type=str,
        default=None,
        help="AES password received after signing the NDA. Not needed with --export-only.",
    )
    parser.add_argument(
        "--download-path",
        type=str,
        default="./data/soccernetballanticipation",
        help="Directory where the dataset is stored (or will be downloaded to).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip download/extraction; only export frames from already-extracted videos.",
    )
    parser.add_argument(
        "--one-split",
        type=str,
        default=None,
        choices=["train", "valid", "test", "challenge"],
        help="Process only one split instead of all.",
    )
    parser.add_argument(
        "--ignore-challenge",
        action="store_true",
        help="Exclude the challenge split.",
    )
    parser.add_argument(
        "--delete-videos",
        action="store_true",
        help="Delete source videos and zip files during export to save storage space.",
    )
    parser.add_argument(
        "--frame-size",
        type=str,
        default="448p",
        choices=["224p", "448p", "720p"],
        help=(
            "Export frames at one of three resolutions:\n"
            "  224p → 398×224\n"
            "  448p → 796×448\n"
            "  720p → 1280×720 (no re-scale)\n"
        ),
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPU workers for parallel frame export.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use CUDA hardware-accelerated decoding in ffmpeg.",
    )
    # ---- Sampling arguments ----
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        metavar="SPLIT:N[,SPLIT:N,…]",
        help=(
            "Randomly sample a fixed number of clips per split before exporting frames.\n"
            "Format: comma-separated 'split:N' pairs, e.g.:\n"
            "  --sample train:200,valid:50,test:50\n"
            "Splits not listed keep all their clips.\n"
            "Non-sampled clip folders are DELETED from disk."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible clip sampling (default: 42).",
    )

    args = parser.parse_args()
    print("Arguments:", args)

    # Resolve the resolution sub-folder name used by the zip layout
    frame_size_path = "224p" if args.frame_size == "224p" else "720p"

    # Validate download key requirement
    if args.download_key is None and not args.export_only:
        raise ValueError("--download-key is required unless --export-only is set.")

    # Build list of splits to process
    if args.one_split is not None:
        splits = [args.one_split]
    elif args.ignore_challenge:
        splits = ["train", "valid", "test"]
    else:
        splits = ["train", "valid", "test", "challenge"]

    # Parse sampling spec
    sample_counts = parse_sample_arg(args.sample)
    if sample_counts:
        unknown = set(sample_counts) - set(splits)
        if unknown:
            print(f"Warning: --sample contains split names not in the processing list: {unknown}")

    split_root = os.path.join(args.download_path, frame_size_path)
    print(f"\nProcessing splits {splits}  |  split root: {split_root}\n")

    for split in splits:
        print(f"{'='*60}")
        print(f"Split: {split}")
        print(f"{'='*60}")

        # 1. Download & extract (unless export-only)
        if not args.export_only:
            download_split(split, args.download_path, frame_size_path)
            extract_split(
                split,
                args.download_key,
                split_root,
                args.delete_videos,
            )

        # 2. Sample clips (delete non-sampled from disk)
        n_samples = sample_counts.get(split)  # None → keep all
        clips = sample_and_prune_clips(split, split_root, n_samples, args.seed)

        # 3. Filter Labels-ball.json when sampling is active for this split
        if n_samples is not None:
            filter_labels_json(split, split_root, clips)

        # 4. Export frames from the (possibly pruned) clip set
        export_frames(
            split,
            split_root,
            clips,
            delete_videos=args.delete_videos,
            frame_size=args.frame_size,
            num_cpus=args.cpus,
            use_cuda=args.use_cuda,
        )

    print("\nDone.")
