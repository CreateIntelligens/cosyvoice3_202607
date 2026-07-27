#!/usr/bin/env python3
"""Validate CosyVoice Kaldi-style manifests and optional parquet outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


SPLITS = ("train", "dev", "test")


def read_keyed_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_number}: expected KEY VALUE")
        key, value = parts
        if key in result:
            raise ValueError(f"{path}:{line_number}: duplicate key {key}")
        result[key] = value
    return result


def validate_split(split_dir: Path, check_audio: bool, require_parquet: bool) -> dict[str, object]:
    wavs = read_keyed_file(split_dir / "wav.scp")
    texts = read_keyed_file(split_dir / "text")
    utt2spk = read_keyed_file(split_dir / "utt2spk")
    instructs = read_keyed_file(split_dir / "instruct")
    expected_keys = set(wavs)
    for name, values in (("text", texts), ("utt2spk", utt2spk), ("instruct", instructs)):
        if set(values) != expected_keys:
            raise ValueError(f"{split_dir}: {name} keys do not match wav.scp")

    spk2utt = read_keyed_file(split_dir / "spk2utt")
    reverse: dict[str, str] = {}
    for speaker, utterances in spk2utt.items():
        for utterance in utterances.split():
            if utterance in reverse:
                raise ValueError(f"{split_dir}: duplicate {utterance} in spk2utt")
            reverse[utterance] = speaker
    if reverse != utt2spk:
        raise ValueError(f"{split_dir}: spk2utt is not the exact reverse mapping of utt2spk")

    total_duration = 0.0
    for utterance, wav_path_text in wavs.items():
        wav_path = Path(wav_path_text)
        if not wav_path.is_file():
            raise FileNotFoundError(f"{split_dir}: missing WAV for {utterance}: {wav_path}")
        source_prefix = utterance.split("-", 1)[0]
        if not utt2spk[utterance].startswith(f"{source_prefix}-"):
            raise ValueError(f"{split_dir}: source/speaker mismatch for {utterance}")
        if not texts[utterance].strip():
            raise ValueError(f"{split_dir}: empty transcript for {utterance}")
        if check_audio:
            total_duration += float(sf.info(wav_path).duration)

    parquet_rows = None
    parquet_columns: list[str] = []
    data_list = split_dir / "parquet" / "data.list"
    if require_parquet:
        if not data_list.is_file():
            raise FileNotFoundError(f"missing parquet data.list: {data_list}")
        shards = [Path(line) for line in data_list.read_text(encoding="utf-8").splitlines() if line]
        if not shards:
            raise ValueError(f"empty parquet data.list: {data_list}")
        parquet_rows = 0
        required_columns = {"utt", "audio_data", "text", "spk", "instruct"}
        for shard in shards:
            if not shard.is_file():
                raise FileNotFoundError(f"missing parquet shard: {shard}")
            metadata = pq.read_metadata(shard)
            parquet_rows += metadata.num_rows
            # ParquetSchema.names may include nested list children named
            # "element". Validate and report top-level Arrow fields only.
            arrow_schema = metadata.schema.to_arrow_schema()
            parquet_columns = [field.name for field in arrow_schema]

            if not required_columns.issubset(parquet_columns):
                raise ValueError(f"{shard}: missing required columns {required_columns - set(parquet_columns)}")
        if parquet_rows != len(wavs):
            raise ValueError(f"{split_dir}: parquet rows {parquet_rows} != manifest rows {len(wavs)}")

    return {
        "utterances": len(wavs),
        "speakers": len(set(utt2spk.values())),
        "hours": round(total_duration / 3600, 6) if check_audio else None,
        "parquet_rows": parquet_rows,
        "parquet_columns": parquet_columns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument("--require-parquet", action="store_true")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    # train/dev/test are mandatory; optional extras such as test_unseen_speaker are
    # validated too so a held-out split cannot silently overlap the training data.
    extra_splits = sorted(
        path.name
        for path in (prepared_dir / "data").iterdir()
        if path.is_dir() and path.name not in SPLITS and (path / "wav.scp").is_file()
    )
    splits = (*SPLITS, *extra_splits)
    summaries: dict[str, object] = {}
    split_keys: dict[str, set[str]] = {}
    for split in splits:
        split_dir = prepared_dir / "data" / split
        summaries[split] = validate_split(split_dir, args.check_audio, args.require_parquet)
        split_keys[split] = set(read_keyed_file(split_dir / "wav.scp"))
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            overlap = split_keys[left] & split_keys[right]
            if overlap:
                raise ValueError(f"{left}/{right} overlap: {sorted(overlap)[:10]}")
    # A held-out split is only meaningful when none of its speakers were trained on.
    for split in extra_splits:
        if not split.startswith("test_unseen_speaker"):
            continue
        held = set(read_keyed_file(prepared_dir / "data" / split / "spk2utt"))
        for trained in ("train", "dev"):
            leaked = held & set(read_keyed_file(prepared_dir / "data" / trained / "spk2utt"))
            if leaked:
                raise ValueError(f"{split} speakers also appear in {trained}: {sorted(leaked)[:10]}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
