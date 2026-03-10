#!/usr/bin/env python3
"""
Create simulstream audio definitions (YAML + references + transcripts) from a NeMo manifest.

Usage:
    python scripts/create_audio_definitions_from_manifest.py \\
        --manifest /lustre/fsw/portfolios/convai/users/lgrigoryan/iwslt26/data/acl_6060/dev/manifests/manifest_en_to_ru.jsonl \\
        --output-dir /path/to/output

Outputs in --output-dir:
    - audio_definitions.yaml   (wav, offset, duration per segment)
    - references.txt           (one reference line per segment)
    - transcripts.txt          (one source transcript per segment)

If duration is missing or 0 in the manifest, the script can optionally compute it from the
WAV file (requires soundfile: pip install soundfile).
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def get_duration_from_wav(wav_path: str) -> float:
    """Read duration in seconds from WAV file. Requires soundfile."""
    try:
        import soundfile as sf
        info = sf.info(wav_path)
        return float(info.duration)
    except Exception as e:
        print(f"Warning: could not read duration from {wav_path}: {e}", file=sys.stderr)
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Create simulstream audio definitions from a NeMo manifest JSONL"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest JSONL (e.g. manifest_en_to_ru.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where audio_definitions.yaml, references.txt, transcripts.txt will be written",
    )
    parser.add_argument(
        "--fill-duration",
        action="store_true",
        help="If duration is 0 or missing, read it from the WAV file (requires soundfile)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_defs = []
    references = []
    transcripts = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            audio_path = data["audio_filepath"]
            duration = data.get("duration", 0.0)
            if (duration is None or duration == 0.0) and args.fill_duration:
                duration = get_duration_from_wav(audio_path)
            duration = float(duration) if duration else 0.0

            offset = data.get("offset", 0.0)
            audio_defs.append({
                "wav": data["id"],
                "offset": offset,
                "duration": duration,
            })

            transcripts.append(data.get("text", ""))
            # target: answer (acl style) or target_text
            ref = data.get("answer", data.get("target_text", ""))
            references.append(ref)

    # Write audio_definitions.yaml
    audio_def_file = output_dir / "audio_definitions.yaml"
    with open(audio_def_file, "w", encoding="utf-8") as f:
        yaml.dump(audio_defs, f, default_flow_style=False, allow_unicode=True)
    print(f"Created: {audio_def_file} ({len(audio_defs)} segments)")

    # Write references.txt
    refs_file = output_dir / "references.txt"
    with open(refs_file, "w", encoding="utf-8") as f:
        for ref in references:
            f.write(ref + "\n")
    print(f"Created: {refs_file}")

    # Write transcripts.txt
    trans_file = output_dir / "transcripts.txt"
    with open(trans_file, "w", encoding="utf-8") as f:
        for trans in transcripts:
            f.write(trans + "\n")
    print(f"Created: {trans_file}")


if __name__ == "__main__":
    main()
