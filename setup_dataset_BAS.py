import argparse
import zipfile
import os
import subprocess
from pathlib import Path
from SoccerNet.utils import getListGames
from SoccerNet.Downloader import SoccerNetDownloader

TASK_NAME = "spotting-ball-2024"

"""
Data needed per game:
- 1.9GB for zip file
- 1.9GB for videos and labels
- 2.5GB for small frames
- 7.8GB for medium frames
- 14.4GB for large frames
"""

def download_split(split, download_key, download_path):
    """
    Download the specified split of the dataset.

    Args:
        split (str): The split to download (train, valid, test, or challenge).
        download_key (str): The download key received by signing the NDA.
        download_path (str): The directory where the dataset will be downloaded.

    Returns:
        None
    """
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=download_path)
    print(f"Downloading {split} split")
    split_path = Path(os.path.join(download_path, TASK_NAME, split+".zip"))
    if split_path.is_file():
        print(f"Split {split} already downloaded. Skipping download.")
    else:
        mySoccerNetDownloader.downloadDataTask(task=TASK_NAME,
                                               split=[split],
                                               password=download_key)

def extract_split(split, download_path, delete_videos=False):
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
    split_path = Path(os.path.join(download_path, TASK_NAME, split+".zip"))
    with zipfile.ZipFile(split_path, "r") as zf:
        zf.extractall(split_path.parent)
    if delete_videos:
        # Delete the zip file after extraction
        os.remove(split_path)
        print(f"Deleted {split} zip file to save space")

def export_frames(split, download_path, delete_videos=False, frame_size="448p"):
    print(f"Exporting frames from {split} split")
    """
    Export frames from the specified split of the dataset using the specified resolution.
    
    Args:
        split (str): The split to export frames from (train, valid, test, or challenge).
        download_path (str): The directory where the dataset is downloaded.
        delete_videos (bool): Whether to delete the downloaded videos after exporting frames.
        frame_size (str): The resolution of the exported frames (224p, 448p, or 720p).

    Returns:
        None
    """
    low_res = frame_size == "224p"
    split_games = getListGames(split, "spotting-ball")
    # Iterate through each game in the split
    for game in split_games:
        # Delete the unneeded video before exporting if deleting
        if delete_videos:
            not_resolution = "720p.mp4" if low_res else "244p.mp4"
            video_path = Path(os.path.join(download_path, TASK_NAME, game, not_resolution))
            if video_path.is_file():
                os.remove(video_path)
                print(f"Deleted {video_path} to save space while exporting frames")
        resolution = "224p.mp4" if low_res else "720p.mp4"
        video_path = Path(os.path.join(download_path, TASK_NAME, game, resolution))
        print(f"Exporting {video_path}")
        # Change the resolution of the video if using 448p. Otherwise, use the original resolution
        if frame_size == "448p":
            subprocess.call(["ffmpeg", "-i", video_path, "-q:v", "1", "-vf", "scale=796x448", os.path.join(video_path.parent, "frame%d.jpg")])
        else:
            subprocess.call(["ffmpeg", "-i", video_path, "-q:v", "1", os.path.join(video_path.parent, "frame%d.jpg")])
        # Delete second video if deleting
        if delete_videos:
            if video_path.is_file():
                os.remove(video_path)
                print(f"Deleted {video_path} to save space")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download ball action spotting dataset and extract frames. Required storage:\n\t" \
    "- 720p without deleting: 18.2GB per game\n\t" \
    "- 720p with deleting:    14.4GB per game\n\t" \
    "- 448p without deleting: 11.6GB per game\n\t" \
    "- 448p with deleting:    7.8GB  per game\n\t" \
    "- 240p without deleting: 6.3GB  per game\n\t" \
    "- 240p with deleting:    2.5GB  per game\n\t",
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
    args = parser.parse_args()
    print("Supplied arguments:", args)

    # Check for required arguments in case of download
    if args.download_key is None and not args.export_only:
        raise ValueError("Download key is required for downloading the dataset.")
    
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
            download_split(split, args.download_key, args.download_path)
            extract_split(split, args.download_path, args.delete_videos)
        export_frames(split, args.download_path, args.delete_videos, args.frame_size)