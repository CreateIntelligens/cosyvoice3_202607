#!/usr/bin/env python3
"""Cut broadcast episode audio into CosyVoice training segments using cue lists.

The broadcast subtitle files record only the *start* timecode of each caption
(``HH:MM:SS:FF text``); the production team never logged when a caption should
disappear, so the naive "next timecode is this caption's end" reading stretches
every window ~56% beyond the actual speech.  Instead this tool trusts only the
start times and recovers the end from the audio: it runs an energy VAD over the
whole episode and keeps the speech that begins at each cue.

Because the source WAVs are vocal-isolated (UVR5 ``back_left``), the silence
between utterances is genuinely silent and the VAD boundaries land within a few
percent of the human-corrected reference.

Output layout is ``OUT_DIR/EPISODE/EPISODE_INDEX.wav`` plus a matching
``.normalized.txt``, which ``prepare_taigi_data.py`` consumes with the episode
acting as the speaker directory (so splits group by episode).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import soundfile as sf


TIMECODE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})[:;](\d{2})\s*(.*)$")
EPISODE_NUM = re.compile(r"(\d+)")
CJK = re.compile(r"[一-鿿 a-zA-Z]".replace(" ", ""))


@dataclass
class Segment:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def char_count(text: str) -> int:
    return len(CJK.findall(text))


def episode_number(path: Path) -> int | None:
    match = EPISODE_NUM.search(path.stem)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- cue list


def parse_cuelist(path: Path, fps: float) -> list[tuple[float, str]]:
    """Return (start_seconds, text) per line; empty text marks a caption clear."""
    for encoding in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")

    cues: list[tuple[float, str]] = []
    for line in raw.splitlines():
        match = TIMECODE.match(line.strip())
        if not match:
            continue
        hours, minutes, seconds, frames, text = match.groups()
        stamp = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(frames) / fps
        text = text.replace("//", " ").strip()
        # A few episodes repeat the timecode inside the text field.
        nested = TIMECODE.match(text)
        if nested:
            text = nested.group(5).replace("//", " ").strip()
        cues.append((stamp, text))
    cues.sort(key=lambda item: item[0])
    return cues


# --------------------------------------------------------------------------- VAD


def load_channel(path: Path, channel: int) -> tuple[np.ndarray, int]:
    """Decode one channel of a full episode.  ~1 GB per hour of 44.1 kHz audio,
    which is worth spending when the source lives on a bandwidth-capped share:
    holding it lets the envelope pass and the cutting pass share a single read."""
    blocks: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as handle:
        sample_rate = handle.samplerate
        for block in handle.blocks(blocksize=1 << 22, dtype="float32", always_2d=True):
            blocks.append(block[:, min(channel, block.shape[1] - 1)].copy())
    if not blocks:
        return np.zeros(0, dtype=np.float32), sample_rate
    return np.concatenate(blocks), sample_rate


def envelope_from_samples(samples: np.ndarray, frame: int) -> np.ndarray:
    usable = (len(samples) // frame) * frame
    if not usable:
        return np.zeros(0, dtype=np.float32)
    framed = samples[:usable].reshape(-1, frame).astype(np.float64)
    return np.sqrt((framed ** 2).mean(axis=1)).astype(np.float32)


def rms_envelope(path: Path, frame: int, channel: int) -> tuple[np.ndarray, int]:
    """Stream the file and return per-frame RMS so 1 GB episodes stay out of RAM."""
    chunks: list[np.ndarray] = []
    with sf.SoundFile(str(path)) as handle:
        sample_rate = handle.samplerate
        leftover = np.zeros(0, dtype=np.float32)
        for block in handle.blocks(blocksize=frame * 4096, dtype="float32", always_2d=True):
            samples = np.concatenate([leftover, block[:, min(channel, block.shape[1] - 1)]])
            usable = (len(samples) // frame) * frame
            if usable:
                framed = samples[:usable].reshape(-1, frame)
                chunks.append(np.sqrt((framed.astype(np.float64) ** 2).mean(axis=1)))
            leftover = samples[usable:]
    if not chunks:
        return np.zeros(0), sample_rate
    return np.concatenate(chunks).astype(np.float32), sample_rate


def speech_runs(
    envelope: np.ndarray,
    sample_rate: int,
    frame: int,
    rel_db: float,
    floor_margin_db: float,
    min_speech: float,
    min_gap: float,
) -> list[list[float]]:
    if envelope.size == 0:
        return []
    peak = float(envelope.max())
    if peak <= 0:
        return []
    db = 20.0 * np.log10(np.maximum(envelope, peak * 1e-6) / peak)
    threshold = max(float(np.percentile(db, 20)) + floor_margin_db, -rel_db)
    voiced = db > threshold

    seconds_per_frame = frame / sample_rate
    runs: list[list[float]] = []
    index = 0
    while index < len(voiced):
        if not voiced[index]:
            index += 1
            continue
        end = index
        while end < len(voiced) and voiced[end]:
            end += 1
        runs.append([index * seconds_per_frame, end * seconds_per_frame])
        index = end

    merged: list[list[float]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < min_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [run for run in merged if run[1] - run[0] >= min_speech]


def speech_within(
    runs: list[list[float]],
    low: float,
    high: float,
    join_gap: float,
    lead: float,
) -> tuple[float, float] | None:
    """Speech that *starts at* ``low``; intersect first so a neighbouring
    utterance's tail can never be mistaken for this cue."""
    pieces: list[tuple[float, float]] = []
    for run_start, run_end in runs:
        start, end = max(run_start, low - lead), min(run_end, high)
        if end - start >= 0.12:
            pieces.append((start, end))
    if not pieces:
        return None

    groups: list[list[tuple[float, float]]] = [[pieces[0]]]
    for piece in pieces[1:]:
        if piece[0] - groups[-1][-1][1] < join_gap:
            groups[-1].append(piece)
        else:
            groups.append([piece])

    reaching = [group for group in groups if group[-1][1] > low + 0.05]
    chosen = reaching[0] if reaching else max(groups, key=lambda g: sum(e - s for s, e in g))
    start, end = chosen[0][0], chosen[-1][1]
    return (start, end) if end - start > 0.25 else None


# --------------------------------------------------------------------------- segmentation


def build_segments(cues: list[tuple[float, str]], runs: list[list[float]], args) -> list[Segment]:
    segments: list[Segment] = []
    for index, (start, text) in enumerate(cues):
        if not text or char_count(text) < args.min_chars:
            continue
        high = cues[index + 1][0] if index + 1 < len(cues) else start + args.tail_window
        high = min(high, start + args.max_window)
        found = speech_within(runs, max(0.0, start), high, args.join_gap, args.lead)
        if found:
            segments.append(Segment(found[0], found[1], text))

    if args.merge_gap > 0:
        merged: list[Segment] = []
        for segment in segments:
            if (
                merged
                and segment.start - merged[-1].end < args.merge_gap
                and segment.end - merged[-1].start <= args.max_duration
            ):
                merged[-1].end = segment.end
                merged[-1].text = f"{merged[-1].text} {segment.text}"
            else:
                merged.append(Segment(segment.start, segment.end, segment.text))
        segments = merged

    kept: list[Segment] = []
    for segment in segments:
        if not args.min_duration <= segment.duration <= args.max_duration:
            continue
        rate = char_count(segment.text) / segment.duration
        if not args.min_cps <= rate <= args.max_cps:
            continue
        kept.append(segment)
    return kept


def write_segments(
    audio_path: Path,
    segments: list[Segment],
    out_dir: Path,
    episode: int,
    channel: int,
    pad_head: float,
    pad_tail: float,
    samples: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    def emit(index: int, segment: Segment, mono: np.ndarray, rate: int) -> bool:
        if len(mono) < int(0.2 * rate):
            return False
        name = f"{episode:04d}_{index:06d}"
        sf.write(str(out_dir / f"{name}.wav"), mono, rate, subtype="PCM_16")
        (out_dir / f"{name}.normalized.txt").write_text(segment.text + "\n", encoding="utf-8")
        return True

    if samples is not None and sample_rate:
        for index, segment in enumerate(segments):
            start = max(0, int((segment.start - pad_head) * sample_rate))
            stop = min(len(samples), int((segment.end + pad_tail) * sample_rate))
            written += emit(index, segment, samples[start:stop], sample_rate)
        return written

    with sf.SoundFile(str(audio_path)) as handle:
        rate = handle.samplerate
        total = len(handle)
        for index, segment in enumerate(segments):
            start = max(0, int((segment.start - pad_head) * rate))
            stop = min(total, int((segment.end + pad_tail) * rate))
            if stop - start < int(0.2 * rate):
                continue
            handle.seek(start)
            block = handle.read(stop - start, dtype="float32", always_2d=True)
            written += emit(index, segment, block[:, min(channel, block.shape[1] - 1)], rate)
    return written


def process_episode(job: dict) -> dict:
    args = argparse.Namespace(**job["args"])
    audio_path = Path(job["audio"])
    cue_path = Path(job["cuelist"])
    episode = job["episode"]

    try:
        cues = parse_cuelist(cue_path, args.fps)
        if not cues:
            return {"episode": episode, "status": "no_cues", "segments": 0, "seconds": 0.0}

        samples = None
        if args.in_memory:
            samples, sample_rate = load_channel(audio_path, args.channel)
            envelope = envelope_from_samples(samples, args.frame)
        else:
            envelope, sample_rate = rms_envelope(audio_path, args.frame, args.channel)
        runs = speech_runs(
            envelope, sample_rate, args.frame, args.rel_db,
            args.floor_margin_db, args.min_speech, args.min_gap,
        )
        segments = build_segments(cues, runs, args)
        seconds = sum(segment.duration for segment in segments)
        rates = [char_count(s.text) / s.duration for s in segments]

        written = 0
        if not args.dry_run:
            written = write_segments(
                audio_path, segments, Path(args.out_dir) / f"{episode:04d}",
                episode, args.channel, args.pad_head, args.pad_tail, samples, sample_rate,
            )
        return {
            "episode": episode,
            "status": "ok",
            "cues": len([c for c in cues if c[1]]),
            "segments": len(segments),
            "written": written,
            "seconds": round(seconds, 2),
            "speech_seconds": round(sum(r[1] - r[0] for r in runs), 2),
            "median_cps": round(float(np.median(rates)), 3) if rates else 0.0,
            "median_duration": round(float(np.median([s.duration for s in segments])), 3) if segments else 0.0,
        }
    except Exception as error:  # noqa: BLE001 - one bad episode must not kill the run
        return {"episode": episode, "status": f"error: {error}", "segments": 0, "seconds": 0.0}


# --------------------------------------------------------------------------- validation


def parse_srt(path: Path) -> list[Segment]:
    for encoding in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")

    pattern = re.compile(
        r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
    )
    out: list[Segment] = []
    for block in re.split(r"\n\s*\n", raw):
        match = pattern.search(block)
        if not match:
            continue
        values = [int(value) for value in match.groups()]
        start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
        end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
        text = "".join(line for line in block.split("\n")[2:] if line.strip()).strip()
        if text and end > start:
            out.append(Segment(start, end, text))
    return out


def validate_episode(job: dict) -> dict:
    args = argparse.Namespace(**job["args"])
    reference = parse_srt(Path(job["reference"]))
    if not reference:
        return {"episode": job["episode"], "status": "no_reference"}

    cues = parse_cuelist(Path(job["cuelist"]), args.fps)
    if args.in_memory:
        samples, sample_rate = load_channel(Path(job["audio"]), args.channel)
        envelope = envelope_from_samples(samples, args.frame)
    else:
        envelope, sample_rate = rms_envelope(Path(job["audio"]), args.frame, args.channel)
    runs = speech_runs(
        envelope, sample_rate, args.frame, args.rel_db,
        args.floor_margin_db, args.min_speech, args.min_gap,
    )
    merge_gap, args.merge_gap = args.merge_gap, 0.0  # compare per cue, not merged
    produced = build_segments(cues, runs, args)
    args.merge_gap = merge_gap

    key = lambda text: re.sub(r"[^一-鿿a-zA-Z]", "", text)
    index: dict[str, list[Segment]] = {}
    for segment in reference:
        index.setdefault(key(segment.text), []).append(segment)

    start_error: list[float] = []
    end_error: list[float] = []
    for segment in produced:
        bucket = index.get(key(segment.text))
        if not bucket:
            continue
        best = min(bucket, key=lambda ref: abs(ref.start - segment.start))
        if abs(best.start - segment.start) > 2.0:
            continue
        start_error.append(segment.start - best.start)
        end_error.append(segment.end - best.end)

    def summarise(values: list[float]) -> dict:
        if not values:
            return {}
        array = np.abs(np.array(values))
        return {
            "median": round(float(np.median(array)), 3),
            "p90": round(float(np.percentile(array, 90)), 3),
            "within_0.1s": round(float((array < 0.1).mean()), 4),
            "within_0.3s": round(float((array < 0.3).mean()), 4),
        }

    ref_seconds = sum(segment.duration for segment in reference)
    our_seconds = sum(segment.duration for segment in produced)
    return {
        "episode": job["episode"],
        "status": "ok",
        "reference_segments": len(reference),
        "produced_segments": len(produced),
        "matched": len(start_error),
        "reference_seconds": round(ref_seconds, 1),
        "produced_seconds": round(our_seconds, 1),
        "duration_ratio": round(our_seconds / ref_seconds, 4) if ref_seconds else 0.0,
        "start_error": summarise(start_error),
        "end_error": summarise(end_error),
    }


# --------------------------------------------------------------------------- driver


def collect_jobs(args) -> list[dict]:
    audio_by_episode: dict[int, Path] = {}
    for path in sorted(Path(args.audio_dir).glob("*.wav")):
        number = episode_number(path)
        if number is not None:
            audio_by_episode[number] = path

    cues_by_episode: dict[int, Path] = {}
    for path in sorted(Path(args.cuelist_dir).iterdir()):
        if path.suffix.lower() not in {".txt", ".srt"} or path.name.lower() == "desktop.ini":
            continue
        number = episode_number(path)
        if number is not None:
            cues_by_episode[number] = path

    wanted = set(audio_by_episode) & set(cues_by_episode)
    if args.episodes:
        selected: set[int] = set()
        for chunk in args.episodes.split(","):
            if "-" in chunk:
                low, high = chunk.split("-")
                selected.update(range(int(low), int(high) + 1))
            else:
                selected.add(int(chunk))
        wanted &= selected

    reference_by_episode: dict[int, Path] = {}
    if args.reference_dir:
        for path in sorted(Path(args.reference_dir).iterdir()):
            if path.suffix.lower() != ".srt":
                continue
            number = episode_number(path)
            if number is not None:
                reference_by_episode[number] = path
        wanted &= set(reference_by_episode)

    payload = {k: v for k, v in vars(args).items() if k not in {"workers", "episodes"}}
    jobs = []
    for episode in sorted(wanted):
        job = {
            "episode": episode,
            "audio": str(audio_by_episode[episode]),
            "cuelist": str(cues_by_episode[episode]),
            "args": payload,
        }
        if reference_by_episode:
            job["reference"] = str(reference_by_episode[episode])
        jobs.append(job)
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, help="directory of full-episode WAVs")
    parser.add_argument("--cuelist-dir", required=True, help="directory of subtitle cue lists")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-dir", default="", help="corrected SRTs; enables validation mode")
    parser.add_argument("--report", default="", help="write the per-episode JSON report here")
    parser.add_argument("--episodes", default="", help="e.g. 1-50,120")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="decode each episode once instead of twice; ~1 GB of RAM per worker "
             "but halves the bytes pulled over a slow share",
    )

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--channel", type=int, default=0, help="back_left is mono duplicated to stereo")
    parser.add_argument("--frame", type=int, default=1024, help="RMS frame in samples")

    parser.add_argument("--rel-db", type=float, default=40.0, help="floor relative to episode peak")
    parser.add_argument("--floor-margin-db", type=float, default=6.0)
    parser.add_argument("--min-speech", type=float, default=0.12)
    parser.add_argument("--min-gap", type=float, default=0.30)
    parser.add_argument("--join-gap", type=float, default=0.35)
    # lead/rel-db defaults were swept against the 170 human-corrected episodes:
    # lead 0.05 centres the onset (median -0.049 s vs -0.197 s at 0.20, so no
    # neighbouring tail is dragged in) and rel-db 40 centres the offset
    # (median +0.001 s).  Clipping speech is worse than keeping silence, so the
    # trailing pad is the larger one.
    parser.add_argument("--lead", type=float, default=0.05)
    parser.add_argument("--pad-head", type=float, default=0.05)
    parser.add_argument("--pad-tail", type=float, default=0.10)

    parser.add_argument("--max-window", type=float, default=20.0)
    parser.add_argument("--tail-window", type=float, default=8.0)
    parser.add_argument("--merge-gap", type=float, default=0.30, help="0 disables merging")
    parser.add_argument("--min-duration", type=float, default=0.8)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--min-cps", type=float, default=3.0)
    parser.add_argument("--max-cps", type=float, default=9.0)
    parser.add_argument("--min-chars", type=int, default=2)

    args = parser.parse_args()
    jobs = collect_jobs(args)
    if not jobs:
        print("no episodes matched", file=sys.stderr)
        return 2

    validating = bool(args.reference_dir)
    worker = validate_episode if validating else process_episode
    print(f"{'validating' if validating else 'segmenting'} {len(jobs)} episodes with {args.workers} workers", flush=True)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, job): job["episode"] for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} episodes", flush=True)

    results.sort(key=lambda item: item["episode"])
    ok = [item for item in results if item.get("status") == "ok"]

    if validating:
        ratios = [item["duration_ratio"] for item in ok if item.get("duration_ratio")]
        starts = [item["start_error"]["median"] for item in ok if item.get("start_error")]
        ends = [item["end_error"]["median"] for item in ok if item.get("end_error")]
        within = [item["end_error"]["within_0.3s"] for item in ok if item.get("end_error")]
        summary = {
            "episodes": len(ok),
            "median_duration_ratio": round(float(np.median(ratios)), 4) if ratios else 0.0,
            "median_start_error": round(float(np.median(starts)), 4) if starts else 0.0,
            "median_end_error": round(float(np.median(ends)), 4) if ends else 0.0,
            "end_within_0.3s": round(float(np.median(within)), 4) if within else 0.0,
            "matched_segments": sum(item.get("matched", 0) for item in ok),
        }
    else:
        seconds = sum(item.get("seconds", 0.0) for item in ok)
        rates = [item["median_cps"] for item in ok if item.get("median_cps")]
        summary = {
            "episodes": len(ok),
            "failed": len(results) - len(ok),
            "segments": sum(item.get("segments", 0) for item in ok),
            "written": sum(item.get("written", 0) for item in ok),
            "hours": round(seconds / 3600, 2),
            "median_cps": round(float(np.median(rates)), 3) if rates else 0.0,
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps({"summary": summary, "episodes": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
