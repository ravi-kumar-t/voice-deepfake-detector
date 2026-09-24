"""
Two-stage Dataset Preparation Script for Hugging Face UncovAI/FOR-2sec.

Features:
- Accesses 'UncovAI/FOR-2sec' on Hugging Face Hub.
- Downloads individual WAV audio files on demand without pulling the full 1.13 GB dataset.
- Supports configurable subset limits via CLI (--real N, --fake N; default: 10 each).
- Deterministic naming: real_0001.wav, fake_0001.wav, etc.
- Validates each audio file with soundfile before marking it as saved.
- Skips already existing valid files.
- Bounded retry/iteration limit (max attempts = requested_count * 3) to prevent infinite loops.
- Terminates cleanly upon reaching the exact target counts.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple
import soundfile as sf
from huggingface_hub import list_repo_files, hf_hub_download

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config


REPO_ID = "UncovAI/FOR-2sec"


def fetch_file_manifest() -> Tuple[List[str], List[str]]:
    """
    Fetches the sorted list of REAL and FAKE audio filenames from the repository.
    """
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    real_files = sorted([f for f in all_files if f.startswith("REAL/") and f.endswith(".wav")])
    fake_files = sorted([f for f in all_files if f.startswith("FAKE/") and f.endswith(".wav")])
    return real_files, fake_files


def is_valid_wav(file_path: Path) -> bool:
    """
    Verifies that the file exists and can be parsed as a valid WAV audio file.
    """
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        info = sf.info(file_path)
        return info.samplerate > 0 and info.channels > 0 and info.frames > 0
    except Exception:
        return False


def download_and_save_samples(
    repo_files: List[str],
    target_count: int,
    dest_dir: Path,
    prefix: str
) -> int:
    """
    Downloads and saves up to target_count audio files with deterministic naming.
    Includes bounded safety attempts (target_count * 3) to prevent infinite loops.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    max_attempts = min(len(repo_files), target_count * 3)

    print(f"[{prefix.upper()}] Target: {target_count} files | Max scan attempts: {max_attempts}")

    for attempt_idx in range(max_attempts):
        if saved_count >= target_count:
            break

        repo_file = repo_files[attempt_idx]
        target_filename = f"{prefix}_{saved_count + 1:04d}.wav"
        target_path = dest_dir / target_filename

        # If already exists and valid, skip re-download
        if is_valid_wav(target_path):
            saved_count += 1
            if saved_count % 5 == 0 or saved_count == target_count:
                print(f"  [{prefix.upper()}] Progress: {saved_count}/{target_count} files (reused existing)")
            continue

        try:
            # Download individual file on demand
            cached_file = hf_hub_download(REPO_ID, filename=repo_file, repo_type="dataset")
            shutil.copy2(cached_file, target_path)

            # Verify integrity
            if is_valid_wav(target_path):
                saved_count += 1
                if saved_count % 5 == 0 or saved_count == target_count or saved_count <= 5:
                    print(f"  [{prefix.upper()}] Progress: {saved_count}/{target_count} -> {target_filename}")
            else:
                if target_path.exists():
                    target_path.unlink()
                print(f"  [{prefix.upper()}] Warning: Invalid audio file encountered at {repo_file}. Skipping.")
        except Exception as err:
            print(f"  [{prefix.upper()}] Error downloading {repo_file}: {err}. Continuing to next.")
            if target_path.exists():
                target_path.unlink()

    return saved_count


def prepare_dataset_subset(num_real: int, num_fake: int):
    """
    Coordinates downloading and verification for REAL and FAKE splits.
    """
    print("=" * 60)
    print(f"FOR-2sec Subset Preparation: {num_real} REAL, {num_fake} FAKE")
    print("=" * 60)

    # 1. Get file lists
    print("\nFetching repository file manifest from Hugging Face...")
    real_repo_files, fake_repo_files = fetch_file_manifest()
    print(f"Repository Manifest: {len(real_repo_files)} REAL files, {len(fake_repo_files)} FAKE files.")

    # 2. Process REAL files
    print(f"\nProcessing REAL files into: {config.REAL_DATA_DIR}")
    real_saved = download_and_save_samples(
        repo_files=real_repo_files,
        target_count=num_real,
        dest_dir=config.REAL_DATA_DIR,
        prefix="real"
    )

    # 3. Process FAKE files
    print(f"\nProcessing FAKE files into: {config.FAKE_DATA_DIR}")
    fake_saved = download_and_save_samples(
        repo_files=fake_repo_files,
        target_count=num_fake,
        dest_dir=config.FAKE_DATA_DIR,
        prefix="fake"
    )

    # 4. Final summary
    total_saved = real_saved + fake_saved
    print("\n" + "=" * 40)
    print("Dataset Preparation Summary:")
    print(f"REAL files saved: {real_saved}")
    print(f"FAKE files saved: {fake_saved}")
    print(f"Total: {total_saved}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced subset from UncovAI/FOR-2sec")
    parser.add_argument("--real", type=int, default=10, help="Number of REAL audio files (default: 10)")
    parser.add_argument("--fake", type=int, default=10, help="Number of FAKE audio files (default: 10)")
    args = parser.parse_args()

    prepare_dataset_subset(
        num_real=args.real,
        num_fake=args.fake
    )


if __name__ == "__main__":
    main()
