#!/usr/bin/env python3
"""Shape the SuíSiann corpus into the layout prepare_taigi_data.py expects.

The point of this corpus is not its size — 4.73 hours from one speaker will not
train anything on its own — but that every sentence carries both spellings:
Han characters and Tâi-lô.  That pairing is what could teach the model to read a
pronunciation annotation, which is the only route to characters it has never
seen.  Enumerating characters does not generalise; the syllable inventory does,
because it is closed at 2,269 tone-bearing syllables.

So each clip is emitted twice, once per spelling, pointing at the same audio:

    hanji/SuiSiann_0001.wav   你敢捌去踅過Uniqlo？
    tailo/SuiSiann_0001.wav   Li2 kam2 bat4 khi3 seh8-kue3 Uniqlo1?

The two live under separate speaker directories because prepare_taigi_data.py
keys transcripts by resolved audio path, so one path can only carry one text.
They are symlinks, and --materialize none references them in place, so the
duplication costs no disk.  Both point at the same voice; the speaker embedding
comes from the audio either way, so the split label is bookkeeping.

Tâi-lô is converted to numeric tones first.  Written with diacritics it does not
survive Qwen's tokenizer — the eighth-tone mark has no precomposed form and
falls back to raw bytes — which would defeat the whole exercise.  See tools/tailo.py.

    python tools/prepare_suisiann.py --source workspace/suisiann/0.2.1 \
        --out workspace/suisiann_prepared
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tailo import has_fragment_risk, to_numeric  # noqa: E402


# 台文漢字有兩成以上會被 tokenizer 拆成位元組（佮、遮、逐），簡體也沒有對應
# 字形。這裡不做繁簡轉換——那是推論端的事，訓練文字保留原樣才對得起音訊。
VARIANTS = ("hanji", "tailo")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="workspace/suisiann/0.2.1",
                        help="解開後的 SuiSiann 目錄，內含 SuiSiann.csv 與 ImTong/")
    parser.add_argument("--out", default="workspace/suisiann_prepared")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS,
                        help="要產生哪幾種寫法；只給一種就是單一寫法的對照組")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()

    import pandas as pd

    source = Path(args.source).resolve()
    frame = pd.read_csv(source / "SuiSiann.csv")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    skipped = {"長度不合": 0, "缺音檔": 0, "轉換失敗": 0, "空白": 0}
    kept_seconds = 0.0

    for variant in args.variants:
        (out / variant).mkdir(exist_ok=True)

    for row in frame.itertuples(index=False):
        wav = source / str(row.音檔)
        duration = float(row.長短)
        if not wav.is_file():
            skipped["缺音檔"] += 1
            continue
        if not args.min_duration <= duration <= args.max_duration:
            skipped["長度不合"] += 1
            continue

        han = " ".join(str(row.漢字).split())
        tailo = to_numeric(" ".join(str(row.羅馬字).split()))
        if not han or not tailo:
            skipped["空白"] += 1
            continue
        # 轉不乾淨就整句丟掉。留著只會讓模型看到位元組碎片，而這正是要避免的。
        if has_fragment_risk(tailo):
            skipped["轉換失敗"] += 1
            continue

        texts = {"hanji": han, "tailo": tailo}
        for variant in args.variants:
            link = out / variant / wav.name
            if link.is_symlink() or link.exists():
                link.unlink()
            # 相對連結，這棵樹才搬得動。絕對路徑在本機沒問題，但傳上叢集之後
            # 每一條都會斷，而斷掉的連結在準備階段才會爆，那時已經排隊很久了。
            link.symlink_to(Path("..") / ".." / wav.relative_to(out.parent))
            records.append({"audio": f"{variant}/{wav.name}", "text": texts[variant]})
        kept_seconds += duration

    index = out / "metadata.jsonl"
    with index.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    clips = len(records) // max(len(args.variants), 1)
    print(f"來源 {len(frame):,} 句")
    for reason, count in skipped.items():
        if count:
            print(f"  略過 {reason}  {count:,}")
    print(f"保留 {clips:,} 段音訊，{kept_seconds / 3600:.2f} 小時")
    print(f"寫法 {', '.join(args.variants)} → {len(records):,} 筆 metadata")
    print(f"\n{index}")
    print(f"傳給 prepare_taigi_data.py：")
    print(f"  --source suisiann={out} --container-source suisiann=<容器內路徑>")
    print(f"  --transcript-jsonl suisiann={index}:{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
