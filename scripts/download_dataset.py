"""
OpenVid-1M Dataset Downloader for Lux Training
================================================
Downloads video parts + CSV metadata from HuggingFace,
extracts videos, and prepares a training-ready dataset.

Dataset: https://huggingface.co/datasets/nkp37/OpenVid-1M
License: CC-BY-4.0 (research and non-commercial)
"""
import os
import sys
import json
import time
import zipfile
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Download OpenVid-1M dataset")
    parser.add_argument(
        "--output_dir", type=str, default="datasets/openvid",
        help="Where to store the dataset"
    )
    parser.add_argument(
        "--num_parts", type=int, default=5,
        help="Number of zip parts to download (each ~40GB)"
    )
    parser.add_argument(
        "--start_part", type=int, default=0,
        help="Starting part index"
    )
    parser.add_argument(
        "--extract", action="store_true", default=True,
        help="Extract zip files after download"
    )
    parser.add_argument(
        "--keep_zips", action="store_true", default=False,
        help="Keep zip files after extraction (saves re-download but uses more space)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="Parallel download workers"
    )
    return parser.parse_args()


def download_file(repo_id, filename, local_dir, desc=""):
    """Download a single file from HuggingFace."""
    from huggingface_hub import hf_hub_download
    
    print(f"  Downloading {filename} {desc}...")
    start = time.perf_counter()
    
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        elapsed = time.perf_counter() - start
        size_gb = os.path.getsize(path) / 1e9
        speed = size_gb / elapsed * 1024 if elapsed > 0 else 0
        print(f"  Done: {filename} ({size_gb:.1f} GB in {elapsed:.0f}s, {speed:.0f} MB/s)")
        return path
    except Exception as e:
        print(f"  FAILED: {filename} - {e}")
        return None


def extract_zip(zip_path, extract_dir):
    """Extract a zip file and optionally delete it."""
    print(f"  Extracting {os.path.basename(zip_path)}...")
    start = time.perf_counter()
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Extract flat (no subdirs) like the OpenVid docs suggest
            for member in zf.namelist():
                filename = os.path.basename(member)
                if not filename:
                    continue
                target = os.path.join(extract_dir, filename)
                with zf.open(member) as source, open(target, 'wb') as target_f:
                    import shutil
                    shutil.copyfileobj(source, target_f)
        
        elapsed = time.perf_counter() - start
        count = len([f for f in os.listdir(extract_dir) if f.endswith('.mp4')])
        print(f"  Extracted {os.path.basename(zip_path)} in {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  Extract FAILED: {e}")
        return False


def build_metadata(video_dir, csv_path, output_path):
    """
    Build a JSONL metadata file matching video files to captions from the CSV.
    Our VideoTextDataset reads JSON/JSONL with {"video": path, "caption": text}.
    """
    import csv as csv_module
    
    print("\n  Building metadata index...")
    
    # Load CSV captions
    caption_map = {}
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_module.DictReader(f)
            columns = reader.fieldnames
            print(f"  CSV columns: {columns}")
            
            # Try to find video and caption columns
            video_col = None
            caption_col = None
            for col in columns:
                cl = col.lower()
                if 'video' in cl or 'path' in cl or 'file' in cl or 'name' in cl:
                    video_col = col
                if 'caption' in cl or 'text' in cl or 'desc' in cl:
                    caption_col = col
            
            if video_col is None:
                video_col = columns[0]
            if caption_col is None:
                caption_col = columns[-1]
            
            print(f"  Using video column: '{video_col}', caption column: '{caption_col}'")
            
            for row in reader:
                vid_name = os.path.basename(row[video_col])
                # Remove extension for flexible matching
                vid_key = os.path.splitext(vid_name)[0]
                caption_map[vid_key] = row[caption_col]
    
    print(f"  Loaded {len(caption_map)} captions from CSV")
    
    # Match to actual video files
    video_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
    ])
    
    matched = 0
    entries = []
    for vf in video_files:
        video_path = os.path.join(video_dir, vf)
        vid_key = os.path.splitext(vf)[0]
        
        caption = caption_map.get(vid_key, "")
        if not caption:
            # Try with different separators
            for k, v in caption_map.items():
                if vid_key in k or k in vid_key:
                    caption = v
                    break
        
        if caption:
            matched += 1
        
        entries.append({
            "video": video_path,
            "caption": caption,
        })
    
    # Write JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"  Metadata: {len(entries)} videos, {matched} with captions")
    print(f"  Saved to: {output_path}")
    return len(entries)


def main():
    args = parse_args()
    
    repo_id = "nkp37/OpenVid-1M"
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    zip_dir = output_dir / "zips"
    
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(zip_dir, exist_ok=True)
    
    print("=" * 60)
    print("  OpenVid-1M Dataset Download")
    print("=" * 60)
    print(f"  Repository:  {repo_id}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Parts:       {args.num_parts} (starting from {args.start_part})")
    print(f"  Est. size:   ~{args.num_parts * 40} GB download")
    print()
    
    # Step 1: Download CSV metadata
    print("[1/4] Downloading metadata CSV...")
    csv_path = download_file(
        repo_id, "data/train/OpenVid-1M.csv", str(output_dir),
        desc="(metadata)"
    )
    
    # Step 2: Get list of available parts
    print("\n[2/4] Finding dataset parts...")
    from huggingface_hub import HfApi
    api = HfApi()
    
    all_files = list(api.list_repo_tree(repo_id, repo_type="dataset"))
    
    # Get zip parts (excluding OpenVidHD parts)
    zip_parts = sorted([
        f.rfilename for f in all_files
        if hasattr(f, 'rfilename')
        and f.rfilename.startswith("OpenVid_part")
        and f.rfilename.endswith(".zip")
    ])
    
    # Select parts to download
    selected = zip_parts[args.start_part:args.start_part + args.num_parts]
    print(f"  Available: {len(zip_parts)} parts")
    print(f"  Selected:  {len(selected)} parts")
    for p in selected:
        print(f"    - {p}")
    
    # Step 3: Download video zips
    print(f"\n[3/4] Downloading {len(selected)} video parts...")
    total_start = time.perf_counter()
    
    for i, part_name in enumerate(selected):
        print(f"\n  --- Part {i+1}/{len(selected)} ---")
        zip_path = download_file(
            repo_id, part_name, str(zip_dir),
            desc=f"({i+1}/{len(selected)})"
        )
        
        if zip_path and args.extract:
            # Find the actual downloaded file
            actual_path = os.path.join(zip_dir, part_name)
            if os.path.exists(actual_path):
                extract_zip(actual_path, str(video_dir))
                if not args.keep_zips:
                    os.remove(actual_path)
                    print(f"  Deleted zip: {part_name}")
    
    total_elapsed = time.perf_counter() - total_start
    print(f"\n  Download + extract completed in {total_elapsed/60:.1f} minutes")
    
    # Step 4: Build metadata
    print("\n[4/4] Building training metadata...")
    metadata_path = output_dir / "metadata.jsonl"
    
    # Find the CSV file
    csv_file = None
    for root, dirs, files in os.walk(str(output_dir)):
        for f in files:
            if f == "OpenVid-1M.csv":
                csv_file = os.path.join(root, f)
                break
    
    num_videos = build_metadata(str(video_dir), csv_file, str(metadata_path))
    
    # Summary
    video_count = len([f for f in os.listdir(str(video_dir)) if f.endswith('.mp4')]) if video_dir.exists() else 0
    total_size_gb = sum(
        os.path.getsize(os.path.join(str(video_dir), f))
        for f in os.listdir(str(video_dir))
        if os.path.isfile(os.path.join(str(video_dir), f))
    ) / 1e9 if video_dir.exists() else 0
    
    print("\n" + "=" * 60)
    print("  Download Complete!")
    print("=" * 60)
    print(f"  Videos:     {video_count}")
    print(f"  Size:       {total_size_gb:.1f} GB")
    print(f"  Metadata:   {metadata_path}")
    print(f"  Video dir:  {video_dir}")
    print()
    print("  To train, run:")
    print(f"    python scripts/train.py \\")
    print(f"      --data_dir {video_dir} \\")
    print(f"      --metadata_file {metadata_path} \\")
    print(f"      --model_variant S \\")
    print(f"      --resolution 256 \\")
    print(f"      --num_frames 16 \\")
    print(f"      --batch_size 1")
    print("=" * 60)


if __name__ == "__main__":
    main()
