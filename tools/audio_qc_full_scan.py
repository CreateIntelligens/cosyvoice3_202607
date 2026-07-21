#!/usr/bin/env python3
"""Read-only full-dataset WAV quality scan used by reports/audio_qc_*.json."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import struct
import sys
import time
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


NEAR_ZERO = 1e-4
ACTIVE_DBFS = -50.0
FRAME_MS = 20


def pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (b[:, 0].astype(np.int32) |
                  (b[:, 1].astype(np.int32) << 8) |
                  (b[:, 2].astype(np.int32) << 16))
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


def suggested_action(flags: list[str]) -> str:
    if "active_audio_lt_20pct" in flags:
        return "人工確認字幕與有效語音；若內容正確，優先重新裁切，不要直接自動刪除"
    if "trailing_silence_gt_0.5s" in flags:
        return "人工確認後修剪尾端靜音，保留約 0.1–0.25 秒"
    if "rms_below_-40dbfs" in flags:
        return "人工確認音質與字幕後，再以資料集層級的一致策略調整音量"
    return "人工複核"


def scan_one(task: tuple[str, str, bool]) -> tuple:
    wav_path_text, dataset_root_text, flag_low_rms = task
    wav_path = Path(wav_path_text)
    root = Path(dataset_root_text)
    try:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()
            compression = wf.getcomptype()
            raw = wf.readframes(frame_count)
        if compression != "NONE":
            raise ValueError(f"unsupported WAV compression: {compression}")
        samples = pcm_to_float(raw, sample_width)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
        if samples.size == 0 or sample_rate <= 0:
            raise ValueError("empty audio or invalid sample rate")

        absolute = np.abs(samples)
        near_zero = absolute < NEAR_ZERO
        near_zero_fraction = float(np.mean(near_zero))
        peak = float(np.max(absolute))
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))

        active_threshold = 10.0 ** (ACTIVE_DBFS / 20.0)
        frame_size = max(1, round(sample_rate * FRAME_MS / 1000))
        complete_frames = samples.size // frame_size
        if complete_frames:
            framed = samples[: complete_frames * frame_size].reshape(complete_frames, frame_size)
            frame_rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1))
            active_frame_ratio = float(np.mean(frame_rms > active_threshold))
        else:
            active_frame_ratio = float(rms > active_threshold)

        nonzero_indices = np.flatnonzero(~near_zero)
        if nonzero_indices.size:
            leading = float(nonzero_indices[0] / sample_rate)
            trailing = float((samples.size - 1 - nonzero_indices[-1]) / sample_rate)
        else:
            leading = trailing = float(samples.size / sample_rate)

        flags = []
        if near_zero_fraction > 0.50:
            flags.append("near_zero_gt_50pct")
        if near_zero_fraction > 0.80:
            flags.append("near_zero_gt_80pct")
        if active_frame_ratio < 0.20:
            flags.append("active_audio_lt_20pct")
        if trailing > 0.50:
            flags.append("trailing_silence_gt_0.5s")
        if flag_low_rms and rms_dbfs < -40.0:
            flags.append("rms_below_-40dbfs")

        duration = float(samples.size / sample_rate)
        if not flags:
            return ("ok", duration, ())

        relative = wav_path.relative_to(root)
        transcript_path = wav_path.with_suffix(".normalized.txt")
        transcript = transcript_path.read_text(encoding="utf-8").strip() if transcript_path.exists() else None
        dataset = root.name
        speaker = relative.parts[0] if len(relative.parts) > 1 else wav_path.stem.split("_")[0]
        record = {
            "relative_wav_path": relative.as_posix(),
            "absolute_wav_path": str(wav_path),
            "relative_transcript_path": transcript_path.relative_to(root).as_posix(),
            "current_speaker_id": speaker,
            "proposed_speaker_id": f"{dataset}-{speaker}",
            "transcript_zh": transcript,
            "duration_sec": round(duration, 6),
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "sample_width_bits": sample_width * 8,
            "peak_amplitude": round(peak, 8),
            "rms_dbfs": round(rms_dbfs, 4),
            "near_zero_fraction": round(near_zero_fraction, 6),
            "active_frame_ratio": round(active_frame_ratio, 6),
            "leading_nearzero_sec": round(leading, 6),
            "trailing_nearzero_sec": round(trailing, 6),
            "approx_trailing_silence_start_sec": round(duration - trailing, 6) if trailing else None,
            "flags": flags,
            "suggested_action": suggested_action(flags),
        }
        return ("flagged", duration, record)
    except Exception as exc:  # preserve the path and continue the full scan
        return ("error", 0.0, {"absolute_wav_path": str(wav_path), "error": repr(exc)})


def wav_paths(root: Path):
    for directory, names, files in os.walk(root):
        names.sort()
        for name in sorted(files):
            if name.lower().endswith(".wav"):
                yield str(Path(directory) / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--flag-low-rms", action="store_true", help="also flag RMS below -40 dBFS")
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        parser.error(f"dataset root does not exist: {root}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    counts = Counter()
    category_counts = Counter()
    flagged_records = []
    errors = []
    total_duration = 0.0
    tasks = ((path, str(root), args.flag_low_rms) for path in wav_paths(root))

    with mp.Pool(processes=args.workers) as pool:
        for index, (status, duration, payload) in enumerate(
            pool.imap_unordered(scan_one, tasks, chunksize=16), start=1
        ):
            counts[status] += 1
            total_duration += duration
            if status == "flagged":
                flagged_records.append(payload)
                category_counts.update(payload["flags"])
            elif status == "error":
                errors.append(payload)
            if args.progress_every and index % args.progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"scanned={index:,} flagged={counts['flagged']:,} errors={counts['error']:,} "
                    f"rate={index / max(elapsed, 1):.1f} files/s",
                    file=sys.stderr,
                    flush=True,
                )

    flagged_records.sort(key=lambda item: item["relative_wav_path"])
    errors.sort(key=lambda item: item["absolute_wav_path"])
    total = sum(counts.values())
    elapsed = time.monotonic() - started
    criteria = {
        "near_zero_gt_50pct": "near_zero_fraction > 0.50",
        "near_zero_gt_80pct": "near_zero_fraction > 0.80；同時也會命中 >50%",
        "active_audio_lt_20pct": "active_frame_ratio < 0.20",
        "trailing_silence_gt_0.5s": "trailing_nearzero_sec > 0.50",
    }
    if args.flag_low_rms:
        criteria["rms_below_-40dbfs"] = "rms_dbfs < -40"
    report = {
        "report_version": 2,
        "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
        "dataset": root.name,
        "dataset_root": str(root),
        "nas_modified": False,
        "scope_note": "全資料集 WAV 的唯讀掃描；flagged_records 是所有命中條件片段的完整清單。",
        "scan": {
            "workers": args.workers,
            "elapsed_sec": round(elapsed, 3),
            "wav_count": total,
            "decoded_ok_count": counts["ok"] + counts["flagged"],
            "decode_error_count": counts["error"],
            "total_audio_hours": round(total_duration / 3600.0, 4),
        },
        "metric_definitions": {
            "near_zero_fraction": "abs(sample) < 1e-4 的樣本比例",
            "active_frame_ratio": "20ms 分框後，RMS > -50 dBFS 的 frame 比例",
            "trailing_nearzero_sec": "最後一個 abs(sample) >= 1e-4 樣本到檔尾的近零時長",
            "rms_dbfs": "全檔 RMS，以 full scale=1.0 換算 dBFS",
        },
        "criteria": criteria,
        "summary": {
            "scanned_wav_count": total,
            "flagged_union_count": len(flagged_records),
            "category_counts": dict(sorted(category_counts.items())),
        },
        "flagged_records": flagged_records,
        "scan_errors": errors,
    }
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output_json)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
