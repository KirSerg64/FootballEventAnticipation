import argparse
import pyzipper
import os
import subprocess
import multiprocessing as mp
from itertools import repeat
from pathlib import Path
from huggingface_hub import snapshot_download

def download_split(split, download_path, frame_size="224p"):
    """
    Download the specified split of the dataset.

    Args:
        split (str): The split to download (train, valid, test, or challenge).
        download_key (str): The download key received by signing the NDA.
        download_path (str): The directory where the dataset will be downloaded.

    Returns:
        None
    """
    print(f"Downloading {split} split")
    split_path = Path(os.path.join(download_path, frame_size, split+".zip"))
    if split_path.is_file():
        print(f"Split {split} already downloaded. Skipping download.")
    else:
        snapshot_download(repo_id="SoccerNet/ActionAnticipation",
                                repo_type="dataset", revision="main",
                                local_dir=download_path,
                                allow_patterns=[f"{frame_size}/*"+split+".zip"])

def extract_split(split, download_key, download_path, delete_videos=False):
    """
    Extract the specified split of the dataset.

    Args:
        split (str): The split to extract (train, valid, test, or challenge).
        download_path (str): The directory where the dataset is downloaded.
        delete_videos (bool): Whether to delete the downloaded zip file after extraction.

    Returns:
        None
    """
    print(f"Extracting {split} split")
    split_path = Path(os.path.join(download_path, split+".zip"))
    with pyzipper.AESZipFile(split_path, "r") as zf:
        zf.extractall(split_path.parent, pwd=download_key.encode())
    if delete_videos:
        # Delete the zip file after extraction
        os.remove(split_path)
        print(f"Deleted {split} zip file to save space")

def export_clip(clip, delete_videos, low_res, download_path, frame_size):
    """
    Export frames from a single clip using the specified resolution.
    
    Args:
        clip (str): The folder name of the clip to export frames from.
        delete_videos (bool): Whether to delete the downloaded videos after exporting frames.
        low_res (bool): Whether the clip is in 224p or 720p resolution.
        download_path (str): The directory where the dataset is downloaded.
        frame_size (str): The resolution of the exported frames (224p, 448p, or 720p).

    Returns:
        None
    """
    resolution = "224p.mp4" if low_res else "720p.mp4"
    video_path = Path(os.path.join(download_path, split, clip, resolution))
    print(f"Exporting {video_path}")
    # Change the resolution of the video if using 448p. Otherwise, use the original resolution
    if frame_size == "448p":
        subprocess.call(["ffmpeg", "-i", video_path, "-q:v", "1", "-vf", "scale=796x448", os.path.join(video_path.parent, "frame%d.jpg")])
    else:
        subprocess.call(["ffmpeg", "-i", video_path, "-q:v", "1", os.path.join(video_path.parent, "frame%d.jpg")])
    # Delete video if deleting
    if delete_videos:
        if video_path.is_file():
            os.remove(video_path)
            print(f"Deleted {video_path} to save space")

def export_frames(split, download_path, delete_videos=False, frame_size="448p", num_cpus=4):
    """
    Export frames from the specified split of the dataset using the specified resolution.
    Exports multiple clips in parallel
    
    Args:
        split (str): The split to export frames from (train, valid, test, or challenge).
        download_path (str): The directory where the dataset is downloaded.
        delete_videos (bool): Whether to delete the downloaded videos after exporting frames.
        frame_size (str): The resolution of the exported frames (224p, 448p, or 720p).
        num_cpus (int): The number of parallel tasks to use to extract clips

    Returns:
        None
    """
    print(f"Exporting frames from {split} split")
    low_res = frame_size == "224p"
    try:
        clips = next(os.walk(os.path.join(download_path, split)))[1]
    except StopIteration:
        print(f"Could not find anything to export in the path {os.path.join(download_path, split)}")
        return
    with mp.Pool(num_cpus) as p:
        p.starmap(export_clip, zip(clips, repeat(delete_videos), repeat(low_res), repeat(download_path), repeat(frame_size)))
        


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download ball action spotting dataset and extract frames. Required storage with deleting:\n\t" \
    "- 720p: train 323GB; valid 85GB; test 164GB; challenge 153GB; All: 725GB\n\t" \
    "- 448p: train 175GB; valid 46GB; test 89GB ; challenge 82GB ; All: 393GB\n\t" \
    "- 240p: train 57GB ; valid 15GB; test 29GB ; challenge 27GB ; All: 128GB\n\t",
    formatter_class= argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--download-key",
        type=str,
        default=None,
        help="The download key received by signing the NDA. No needed if export-only",
    )
    parser.add_argument(
        "--download-path",
        type=str,
        default="./data/soccernetball",
        help="Directory where the dataset is to be downloaded in, or is stored in already",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export frames from the pre-downloaded videos. Expects zip files to be in the download path",
    )
    parser.add_argument(
        "--one-split",
        type=str,
        default=None,
        choices=["train", "valid", "test", "challenge"],
        help="Only download and/or export one split",
    )
    parser.add_argument(
        "--ignore-challenge",
        action="store_true",
        help="Ignore the challenge set when downloading and/or exporting",
    )
    parser.add_argument(
        "--delete-videos",
        action="store_true",
        help="Downloads one split at a time and deletes the videos and zip file while exporting to save storage space",
    )
    parser.add_argument(
        "--frame-size",
        type=str,
        default="448p",
        choices=["224p", "448p", "720p"],
        help="Export frames in one of three sizes:\n\t" \
        " - low resolution video  398x224  (224p)\n\t" \
        " - mid resolution video  796x448  (448p)\n\t" \
        " - high resolution video 1280x720 (720p)\n\t",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPUs to use when exporting clips"

    )
    args = parser.parse_args()
    print("Supplied arguments:", args)
    frame_size_path = "224p" if args.frame_size == "224p" else "720p"

    # Check for required arguments in case of download
    if args.download_key is None and not args.export_only:
        raise ValueError("Download key is required for unzipping the dataset.")
    
    # Create list of splits
    if args.one_split is not None:
        if args.one_split not in ["train", "valid", "test", "challenge"]:
            raise ValueError(f"Invalid split name {args.one_split}. Choose from train, valid, test, or challenge.")
        splits = [args.one_split]
    else:
        splits = ["train", "valid", "test"] if args.ignore_challenge else ["train", "valid", "test", "challenge"]

    # Start downloading and exporting
    print(f"Downloading splits: {splits} to the path: {args.download_path}")
    for split in splits:
        if not args.export_only:
            download_split(split, args.download_path, frame_size_path)
            extract_split(split, args.download_key, os.path.join(args.download_path, frame_size_path), args.delete_videos)
        export_frames(split, os.path.join(args.download_path, frame_size_path), args.delete_videos, args.frame_size, args.cpus)