#!/usr/bin/env python3
"""Prepare deterministic CosyVoice manifests from multi-source Taigi WAV trees.

The source layout is expected to be SOURCE_ROOT/SPEAKER/GROUP/*.wav with a
matching *.normalized.txt beside each WAV.  Source names are added to utterance
and speaker IDs so identical numeric IDs in drama1/drama2 never collide.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import tempfile

import soundfile as sf


INSTRUCT = "You are a helpful assistant.<|endofprompt|>"


@dataclass(frozen=True)
class Candidate:
    source: str
    source_root: Path
    relative_wav: Path
    transcript_path: Path
    transcript: str
    speaker_id: str
    utterance_id: str
    group_id: str
    duration_sec: float


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def parse_mapping(values: list[str], option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects NAME=VALUE, got: {value}")
        name, mapped = value.split("=", 1)
        if not name or not mapped:
            raise ValueError(f"{option} expects non-empty NAME=VALUE, got: {value}")
        if name in result:
            raise ValueError(f"duplicate {option} name: {name}")
        result[name] = mapped
    return result


def parse_speaker_lists(values: list[str]) -> dict[str, list[str]]:
    parsed = parse_mapping(values, "--speaker-list")
    return {name: [item for item in value.split(",") if item] for name, value in parsed.items()}


def load_qc_exclusions(values: list[str]) -> dict[str, set[str]]:
    reports = parse_mapping(values, "--qc-report")
    exclusions: dict[str, set[str]] = {}
    for source, report_path in reports.items():
        path = Path(report_path)
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        exclusions[source] = {
            record["relative_wav_path"]
            for record in report.get("flagged_records", [])
            if record.get("relative_wav_path")
        }
    return exclusions


def choose_speakers(
    source: str,
    root: Path,
    explicit: dict[str, list[str]],
    speakers_per_source: int,
) -> list[str]:
    available = sorted((path.name for path in root.iterdir() if path.is_dir()), key=natural_key)
    selected = explicit.get(source, available)
    missing = sorted(set(selected) - set(available), key=natural_key)
    if missing:
        raise ValueError(f"{source} missing speaker directories: {missing}")
    if source not in explicit and speakers_per_source > 0:
        selected = selected[:speakers_per_source]
    if not selected:
        raise ValueError(f"{source} has no selected speakers")
    return selected


def scan_candidates(
    source: str,
    root: Path,
    speakers: list[str],
    excluded: set[str],
    min_duration: float,
    max_duration: float,
) -> tuple[list[Candidate], dict[str, int]]:
    candidates: list[Candidate] = []
    counters: defaultdict[str, int] = defaultdict(int)
    seen_utterances: set[str] = set()
    for raw_speaker in speakers:
        speaker_root = root / raw_speaker
        for wav_path in sorted(speaker_root.rglob("*.wav")):
            counters["wav_seen"] += 1
            relative_wav = wav_path.relative_to(root)
            relative_key = relative_wav.as_posix()
            if relative_key in excluded:
                counters["qc_excluded"] += 1
                continue
            transcript_path = wav_path.with_suffix(".normalized.txt")
            if not transcript_path.is_file():
                counters["missing_transcript"] += 1
                continue
            transcript = " ".join(transcript_path.read_text(encoding="utf-8").split())
            if not transcript:
                counters["empty_transcript"] += 1
                continue
            try:
                duration_sec = float(sf.info(wav_path).duration)
            except Exception:
                counters["audio_info_error"] += 1
                continue
            if duration_sec < min_duration or duration_sec > max_duration:
                counters["duration_excluded"] += 1
                continue

            speaker_id = f"{source}-{raw_speaker}"
            utterance_id = f"{source}-{wav_path.stem}"
            if utterance_id in seen_utterances:
                raise ValueError(f"duplicate utterance ID {utterance_id}; source path: {wav_path}")
            seen_utterances.add(utterance_id)
            parent_parts = relative_wav.parent.parts[1:]
            group_suffix = "/".join(parent_parts) if parent_parts else raw_speaker
            candidates.append(
                Candidate(
                    source=source,
                    source_root=root,
                    relative_wav=relative_wav,
                    transcript_path=transcript_path,
                    transcript=transcript,
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    group_id=f"{source}:{group_suffix}",
                    duration_sec=duration_sec,
                )
            )
            counters["accepted"] += 1
    return candidates, dict(counters)


def split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    counts = [int(total * ratio) for ratio in ratios]
    for index in range(total - sum(counts)):
        counts[index % 3] += 1
    if total >= 3:
        for index in (1, 2):
            if ratios[index] > 0 and counts[index] == 0:
                donor = max(range(3), key=counts.__getitem__)
                counts[donor] -= 1
                counts[index] += 1
    return counts[0], counts[1], counts[2]


def utterance_split(
    candidates: list[Candidate],
    ratios: tuple[float, float, float],
    max_per_speaker: int,
    seed: int,
) -> dict[str, list[Candidate]]:
    by_speaker: defaultdict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_speaker[candidate.speaker_id].append(candidate)
    result = {"train": [], "dev": [], "test": []}
    for speaker_id in sorted(by_speaker, key=natural_key):
        items = sorted(by_speaker[speaker_id], key=lambda item: item.relative_wav.as_posix())
        random.Random(f"{seed}:{speaker_id}").shuffle(items)
        if max_per_speaker > 0:
            items = items[:max_per_speaker]
        train_count, dev_count, _ = split_counts(len(items), ratios)
        result["train"].extend(items[:train_count])
        result["dev"].extend(items[train_count:train_count + dev_count])
        result["test"].extend(items[train_count + dev_count:])
    return result


def group_split(
    candidates: list[Candidate],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Candidate]]:
    groups = sorted({candidate.group_id for candidate in candidates}, key=natural_key)
    random.Random(seed).shuffle(groups)
    train_count, dev_count, _ = split_counts(len(groups), ratios)
    assignment = {
        group: "train" if index < train_count else "dev" if index < train_count + dev_count else "test"
        for index, group in enumerate(groups)
    }
    result = {"train": [], "dev": [], "test": []}
    for candidate in candidates:
        result[assignment[candidate.group_id]].append(candidate)
    return result


def container_wav_path(
    candidate: Candidate,
    materialize: str,
    container_output_root: str,
    container_sources: dict[str, str],
) -> str:
    if materialize == "copy":
        relative_tail = PurePosixPath(*candidate.relative_wav.parts[1:])
        return str(PurePosixPath(container_output_root) / "merged" / candidate.speaker_id / relative_tail)
    return str(PurePosixPath(container_sources[candidate.source]) / candidate.relative_wav.as_posix())


def write_manifests(
    temp_root: Path,
    split_name: str,
    candidates: list[Candidate],
    materialize: str,
    container_output_root: str,
    container_sources: dict[str, str],
) -> None:
    split_dir = temp_root / "data" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(candidates, key=lambda item: item.utterance_id)
    spk2utt: defaultdict[str, list[str]] = defaultdict(list)
    wav_lines, text_lines, utt2spk_lines, instruct_lines = [], [], [], []
    for candidate in ordered:
        wav_path = container_wav_path(candidate, materialize, container_output_root, container_sources)
        if any(character.isspace() for character in wav_path):
            raise ValueError(f"CosyVoice wav.scp paths cannot contain whitespace: {wav_path}")
        wav_lines.append(f"{candidate.utterance_id} {wav_path}\n")
        text_lines.append(f"{candidate.utterance_id} {candidate.transcript}\n")
        utt2spk_lines.append(f"{candidate.utterance_id} {candidate.speaker_id}\n")
        instruct_lines.append(f"{candidate.utterance_id} {INSTRUCT}\n")
        spk2utt[candidate.speaker_id].append(candidate.utterance_id)
    (split_dir / "wav.scp").write_text("".join(wav_lines), encoding="utf-8")
    (split_dir / "text").write_text("".join(text_lines), encoding="utf-8")
    (split_dir / "utt2spk").write_text("".join(utt2spk_lines), encoding="utf-8")
    (split_dir / "instruct").write_text("".join(instruct_lines), encoding="utf-8")
    (split_dir / "spk2utt").write_text(
        "".join(f"{speaker} {' '.join(utterances)}\n" for speaker, utterances in sorted(spk2utt.items())),
        encoding="utf-8",
    )


def materialize_files(temp_root: Path, candidates: list[Candidate], copy_workers: int) -> None:
    def copy_candidate(candidate: Candidate) -> None:
        relative_tail = Path(*candidate.relative_wav.parts[1:])
        destination_wav = temp_root / "merged" / candidate.speaker_id / relative_tail
        destination_txt = destination_wav.with_suffix(".normalized.txt")
        destination_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.source_root / candidate.relative_wav, destination_wav)
        shutil.copy2(candidate.transcript_path, destination_txt)

    with ThreadPoolExecutor(max_workers=copy_workers) as executor:
        list(executor.map(copy_candidate, candidates))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--container-source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--speaker-list", action="append", default=[], metavar="NAME=ID,ID")
    parser.add_argument("--speakers-per-source", type=int, default=0)
    parser.add_argument("--qc-report", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--container-output-root", required=True)
    parser.add_argument("--materialize", choices=("copy", "none"), default="none")
    parser.add_argument("--copy-workers", type=int, default=8)
    parser.add_argument("--split-unit", choices=("utterance", "group"), default="group")
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--dev-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--max-per-speaker", type=int, default=0)
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=1986)
    args = parser.parse_args()

    sources = {name: Path(path).resolve() for name, path in parse_mapping(args.source, "--source").items()}
    container_sources = parse_mapping(args.container_source, "--container-source")
    explicit_speakers = parse_speaker_lists(args.speaker_list)
    exclusions = load_qc_exclusions(args.qc_report)
    ratios = (args.train_ratio, args.dev_ratio, args.test_ratio)
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train/dev/test ratios must be non-negative and sum to 1")
    if args.materialize == "none" and set(container_sources) != set(sources):
        raise ValueError("--materialize none requires one --container-source for every source")
    if args.split_unit == "group" and args.max_per_speaker > 0:
        raise ValueError("group split is intended for full data; do not combine it with --max-per-speaker")
    if args.copy_workers < 1:
        raise ValueError("--copy-workers must be at least 1")

    all_candidates: list[Candidate] = []
    scan_summary: dict[str, object] = {}
    selected_speakers: dict[str, list[str]] = {}
    for source, root in sources.items():
        if not root.is_dir():
            raise FileNotFoundError(f"source root not found: {root}")
        speakers = choose_speakers(source, root, explicit_speakers, args.speakers_per_source)
        selected_speakers[source] = speakers
        candidates, counters = scan_candidates(
            source,
            root,
            speakers,
            exclusions.get(source, set()),
            args.min_duration,
            args.max_duration,
        )
        all_candidates.extend(candidates)
        scan_summary[source] = counters

    if args.split_unit == "utterance":
        split_data = utterance_split(all_candidates, ratios, args.max_per_speaker, args.seed)
    else:
        split_data = group_split(all_candidates, ratios, args.seed)
    if any(not split_data[name] for name in ("train", "dev", "test")):
        raise ValueError("one or more splits are empty; adjust ratios, selected speakers, or split unit")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        if args.materialize == "copy":
            materialize_files(
                temp_root,
                [item for values in split_data.values() for item in values],
                args.copy_workers,
            )
        for split_name, candidates in split_data.items():
            write_manifests(
                temp_root,
                split_name,
                candidates,
                args.materialize,
                args.container_output_root,
                container_sources,
            )
        with (temp_root / "selection.jsonl").open("w", encoding="utf-8") as stream:
            for split_name, candidates in split_data.items():
                for candidate in sorted(candidates, key=lambda item: item.utterance_id):
                    stream.write(json.dumps({
                        "utt": candidate.utterance_id,
                        "speaker": candidate.speaker_id,
                        "split": split_name,
                        "source": candidate.source,
                        "source_relative_wav": candidate.relative_wav.as_posix(),
                        "duration_sec": round(candidate.duration_sec, 6),
                        "group": candidate.group_id,
                    }, ensure_ascii=False) + "\n")
        summary = {
            "seed": args.seed,
            "materialize": args.materialize,
            "split_unit": args.split_unit,
            "ratios": {"train": ratios[0], "dev": ratios[1], "test": ratios[2]},
            "selected_speakers": selected_speakers,
            "scan": scan_summary,
            "splits": {
                name: {
                    "utterances": len(values),
                    "hours": round(sum(item.duration_sec for item in values) / 3600, 6),
                    "speakers": len({item.speaker_id for item in values}),
                }
                for name, values in split_data.items()
            },
        }
        (temp_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_root, output_dir)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
