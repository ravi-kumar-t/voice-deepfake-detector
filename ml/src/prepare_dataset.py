"""
Dataset Preparation Script for ASVspoof (2021 DF / 2019 LA) -> Hackathon Subset.

Functionality:
1. Reads ASVspoof protocol/metadata file (e.g., trial_metadata.txt or keys.txt).
2. Parses labels:
   - 'bonafide' -> REAL (Label 0)
   - 'spoof'    -> FAKE (Label 1)
3. Copies a balanced, manageable subset (default: 500 real + 500 fake) to:
   - ml/data/real/
   - ml/data/fake/
4. Preserves original files and prevents duplicates without loading audio into memory.
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config


def find_asvspoof_files(search_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Attempts to auto-discover trial_metadata.txt/protocol and the flac audio folder.
    
    Returns:
        (protocol_path, flac_dir)
    """
    protocol_path: Optional[Path] = None
    flac_dir: Optional[Path] = None

    # Search for protocol / metadata file
    protocol_candidates = [
        "trial_metadata.txt",
        "keys/DF/CM/trial_metadata.txt",
        "keys/DF/trial_metadata.txt",
        "keys/LA/CM/trial_metadata.txt",
        "ASVspoof2021_DF_eval/trial_metadata.txt",
        "ASVspoof2021_DF_eval/keys/DF/trial_metadata.txt",
        "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
        "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    ]

    for cand in protocol_candidates:
        p = search_dir / cand
        if p.exists() and p.is_file():
            protocol_path = p
            break

    if not protocol_path and search_dir.exists():
        # Generic search for trial_metadata.txt or *.txt protocol
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.lower() in ["trial_metadata.txt", "cm_eval.txt", "keys.txt"] or "protocol" in f.lower():
                    protocol_path = Path(root) / f
                    break
            if protocol_path:
                break

    # Search for flac audio folder
    flac_candidates = [
        search_dir / "flac",
        search_dir / "ASVspoof2021_DF_eval" / "flac",
        search_dir / "ASVspoof2021_DF_eval_part00" / "flac",
        search_dir / "ASVspoof2019_LA_eval" / "flac",
        search_dir / "ASVspoof2019_LA_train" / "flac",
    ]

    for cand in flac_candidates:
        if cand.exists() and cand.is_dir():
            flac_dir = cand
            break

    if not flac_dir and search_dir.exists():
        for root, dirs, _ in os.walk(search_dir):
            for d in dirs:
                if d.lower() == "flac":
                    flac_dir = Path(root) / d
                    break
            if flac_dir:
                break

    return protocol_path, flac_dir


def parse_protocol_file(protocol_path: Path) -> Tuple[List[str], List[str]]:
    """
    Parses ASVspoof protocol/metadata file to extract bona fide and spoof file keys.
    
    Supports:
    - ASVspoof 2021 DF: [SPEAKER] [FILE_ID] [CODEC] [CHANNEL] [ATTACK] [KEY] [TRIM] [SUBSET]
    - ASVspoof 2019 LA/DF: [SPEAKER] [FILE_ID] [ENVIRONMENT/TRASH] [ATTACK] [KEY]
    
    Returns:
        (bonafide_ids, spoof_ids): Lists of base audio filenames (without extension or with).
    """
    bonafide_ids: List[str] = []
    spoof_ids: List[str] = []

    print(f"[Protocol] Reading metadata from: {protocol_path}")

    with open(protocol_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            file_id = parts[1]  # Standard position for audio file ID in ASVspoof
            
            # Search for 'bonafide' or 'spoof' keyword across row tokens
            tokens_lower = [p.lower() for p in parts]
            if "bonafide" in tokens_lower:
                bonafide_ids.append(file_id)
            elif "spoof" in tokens_lower:
                spoof_ids.append(file_id)

    return bonafide_ids, spoof_ids


def copy_samples(
    file_ids: List[str],
    audio_dir: Path,
    dest_dir: Path,
    target_count: int,
    label_name: str
) -> int:
    """
    Copies up to `target_count` available audio files to `dest_dir`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    missing = 0
    already_present = 0

    for file_id in file_ids:
        if copied >= target_count:
            break

        # Check possible filename variations (.flac, .wav, or raw id)
        candidate_paths = [
            audio_dir / f"{file_id}.flac",
            audio_dir / f"{file_id}.wav",
            audio_dir / file_id
        ]

        source_file = None
        for cand in candidate_paths:
            if cand.exists() and cand.is_file():
                source_file = cand
                break

        if not source_file:
            missing += 1
            continue

        target_file = dest_dir / source_file.name

        if target_file.exists():
            already_present += 1
            copied += 1
            continue

        shutil.copy2(source_file, target_file)
        copied += 1

    print(f"[Copy] {label_name}: Successfully prepared {copied} files in '{dest_dir.relative_to(config.ML_DIR)}' (Missing in audio folder: {missing}).")
    return copied


def prepare_dataset(
    asv_root: Optional[Path] = None,
    protocol_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    num_real: int = 500,
    num_fake: int = 500,
    seed: int = 42
):
    """
    Prepares balanced dataset subset from ASVspoof.
    """
    # 1. Determine paths
    if asv_root is None and (protocol_path is None or audio_dir is None):
        # Check standard default search paths
        candidate_roots = [
            config.ML_DIR / "ASVspoof2021_DF_eval",
            config.ML_DIR / "asvspoof",
            config.ML_DIR / "dataset",
            Path("C:/Users/mopad/Downloads/ASVspoof2021_DF_eval"),
            Path("C:/ASVspoof2021_DF_eval"),
        ]
        for c in candidate_roots:
            if c.exists():
                asv_root = c
                break

    if asv_root and (protocol_path is None or audio_dir is None):
        auto_proto, auto_audio = find_asvspoof_files(asv_root)
        protocol_path = protocol_path or auto_proto
        audio_dir = audio_dir or auto_audio

    # Validate paths
    if not protocol_path or not protocol_path.exists():
        print("=" * 70)
        print("[Error] ASVspoof protocol/metadata file not found.")
        print("Please provide the path using --protocol and --audio-dir flags.")
        print("\nExample:")
        print("  python src/prepare_dataset.py --protocol C:/path/to/trial_metadata.txt --audio-dir C:/path/to/flac")
        print("  or")
        print("  python src/prepare_dataset.py --asv-root C:/path/to/ASVspoof2021_DF_eval")
        print("=" * 70)
        return

    if not audio_dir or not audio_dir.exists():
        print("=" * 70)
        print(f"[Error] Audio directory '{audio_dir}' not found.")
        print("Please provide the path to the 'flac' audio directory using --audio-dir.")
        print("=" * 70)
        return

    print("=" * 70)
    print("ASVspoof -> Deepfake Detection Suite Dataset Subsetting")
    print(f"Protocol File: {protocol_path}")
    print(f"Audio Folder:  {audio_dir}")
    print(f"Target Count:  {num_real} REAL (bona fide), {num_fake} FAKE (spoof)")
    print("=" * 70)

    # 2. Parse protocol
    bonafide_ids, spoof_ids = parse_protocol_file(protocol_path)
    print(f"[Protocol Summary] Discovered {len(bonafide_ids)} bona fide (REAL) and {len(spoof_ids)} spoof (FAKE) entries.")

    # 3. Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(bonafide_ids)
    random.shuffle(spoof_ids)

    # 4. Copy balanced subset
    real_copied = copy_samples(
        file_ids=bonafide_ids,
        audio_dir=audio_dir,
        dest_dir=config.REAL_DATA_DIR,
        target_count=num_real,
        label_name="REAL (bona fide)"
    )

    fake_copied = copy_samples(
        file_ids=spoof_ids,
        audio_dir=audio_dir,
        dest_dir=config.FAKE_DATA_DIR,
        target_count=num_fake,
        label_name="FAKE (spoof)"
    )

    # 5. Final summary
    total = real_copied + fake_copied
    print("\n" + "=" * 40)
    print("Dataset Preparation Summary:")
    print(f"REAL: {real_copied}")
    print(f"FAKE: {fake_copied}")
    print(f"TOTAL: {total}")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced ASVspoof subset for Voice Deepfake Detector")
    parser.add_argument("--asv-root", type=str, default=None, help="Root folder of extracted ASVspoof dataset")
    parser.add_argument("--protocol", "-p", type=str, default=None, help="Path to trial_metadata.txt or protocol file")
    parser.add_argument("--audio-dir", "-a", type=str, default=None, help="Path to folder containing .flac audio files")
    parser.add_argument("--num-real", type=int, default=500, help="Number of real audio samples to copy (default: 500)")
    parser.add_argument("--num-fake", type=int, default=500, help="Number of fake audio samples to copy (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible selection")
    args = parser.parse_args()

    prepare_dataset(
        asv_root=Path(args.asv_root) if args.asv_root else None,
        protocol_path=Path(args.protocol) if args.protocol else None,
        audio_dir=Path(args.audio_dir) if args.audio_dir else None,
        num_real=args.num_real,
        num_fake=args.num_fake,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
