import os
import zipfile
import shutil
from pathlib import Path

def get_dir_size(path):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total

def zip_shards(source_dir, output_prefix, shard_size_gb=1):
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Source directory {source_path} does not exist.")
        return

    # Get all subdirectories (clips)
    clips = [x for x in source_path.iterdir() if x.is_dir()]
    clips.sort()  # Sort for consistency

    current_shard_index = 1
    current_shard_size = 0
    current_shard_clips = []
    shard_size_bytes = shard_size_gb * 1024 * 1024 * 1024

    output_dir = source_path.parent / "shards"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(clips)} clips. Starting compression into {output_dir}...")

    for clip in clips:
        clip_size = get_dir_size(clip)

        # If adding this clip exceeds the limit and we have clips in the buffer, write the shard
        if current_shard_size + clip_size > shard_size_bytes and current_shard_clips:
            shard_name = output_dir / f"{output_prefix}_part_{current_shard_index:03d}.zip"
            print(f"Creating {shard_name} ({current_shard_size / (1024*1024):.2f} MB)...")

            with zipfile.ZipFile(shard_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                for clip_path in current_shard_clips:
                    # Add clip folder and its contents to zip, preserving relative path inside 'train'
                    for root, dirs, files in os.walk(clip_path):
                        for file in files:
                            file_path = Path(root) / file
                            # Arcname should be relative to the source_dir's parent to keep 'train/clip_X'
                            arcname = file_path.relative_to(source_path)
                            zf.write(file_path, arcname)

            # Delete original clips after successful zipping
            print(f"Deleting {len(current_shard_clips)} original clips...")
            for clip_path in current_shard_clips:
                shutil.rmtree(clip_path)

            current_shard_index += 1
            current_shard_size = 0
            current_shard_clips = []

        current_shard_clips.append(clip)
        current_shard_size += clip_size

    # Write the last shard if there are remaining clips
    if current_shard_clips:
        shard_name = output_dir / f"{output_prefix}_part_{current_shard_index:03d}.zip"
        print(f"Creating {shard_name} ({current_shard_size / (1024*1024):.2f} MB)...")
        with zipfile.ZipFile(shard_name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for clip_path in current_shard_clips:
                for root, dirs, files in os.walk(clip_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(source_path)
                        zf.write(file_path, arcname)

        # Delete original clips after successful zipping
        print(f"Deleting {len(current_shard_clips)} original clips...")
        for clip_path in current_shard_clips:
            shutil.rmtree(clip_path)

    print("Compression finished.")


if __name__ == "__main__":
#add parameters for cmdline execution
    import argparse
    parser = argparse.ArgumentParser(description="Shard and compress data into zip files.")
    parser.add_argument("source_directory", type=str, help="Path to the source directory containing the clips (e.g., 'train').")
    parser.add_argument("output_prefix", type=str, help="Prefix for the output zip files (e.g., 'train_shard').")
    parser.add_argument("--shard_size_gb", type=float, default=1.0, help="Maximum size of each shard in GB (default: 1.0).")
    args = parser.parse_args()  
    
    zip_shards(args.source_directory, args.output_prefix, args.shard_size_gb)