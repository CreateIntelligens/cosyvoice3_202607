#!/usr/bin/env python3
"""Build the dual-script view of the naer (TAT-MOE) corpus.

naer 的每一筆 JSON 本來就帶 `台羅數字調`——107,294 句、192.6 小時，一句不缺。
這是手上最大的一份漢字／台羅平行語料，比媠聲多 42 倍，而且**沒有授權問題**：
它早就是 v2/v4/v6 的訓練資料。

    "漢羅台文":   "紲落來看新竹市明仔載二十六號的天氣"
    "台羅數字調": "sua3-loh8 lai5-khuann3 sin1-tik4-tshi7 ..."

要注意的差別：naer 的**音訊已經在 v6 的底裡**（配漢字那一側）。所以第二階段拿它
來訓，不是加新內容，是替同一批聲音補上第二種寫法。模型會第二次看到同樣的音訊，
過擬合的風險比媠聲那次高——媠聲不在底裡。

值不值得就是要測出來的：v9 = v6底 + 媠聲 + naer台羅，對照 v8 = v6底 + 媠聲 + 教育部。
v8 的教育部資料是 CC BY-NC-ND 不能出貨，v9 完全乾淨。**兩者若打平，就不必等
教育部回信。**

`台羅數字調` 已經是數字調，不需要再過 tools/tailo.py，但仍然檢查碎片：欄位裡偶爾
混著全形括號與註記，轉不乾淨的整筆丟掉。

    python tools/prepare_naer_tailo.py --out workspace/naer_tailo_prepared
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tailo import has_fragment_risk  # noqa: E402


PARTS = ("TAT-MOE-train", "TAT-MOE-eval", "TAT-MOE-test")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source",
                        default="/mnt/nas/ml-material/naer_corpus/origin_data/1mic_version")
    parser.add_argument("--out", default="workspace/naer_tailo_prepared")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--tailo-only", action="store_true",
                        help="只產台羅那一側。漢字那側已經在底模的訓練資料裡，"
                             "重複餵可能加重過擬合——想試就開這個。")
    args = parser.parse_args()

    root = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    variants = ("tailo",) if args.tailo_only else ("hanji", "tailo")
    for variant in variants:
        (out / variant).mkdir(exist_ok=True)
    audio_root = out / "_audio"
    audio_root.mkdir(exist_ok=True)

    records: list[dict[str, str]] = []
    skipped = {"缺音檔": 0, "缺文字": 0, "台羅有碎片": 0, "長度不合": 0}
    kept = 0
    hours = 0.0

    for part in PARTS:
        jdir = root / f"{part}-json"
        if not jdir.is_dir():
            continue
        # 兩份拷貝的音檔目錄長得不一樣：NAS 上是 <part>-lavalier/lavalier/，
        # Taipei-1 上是 <part>/lavalier/。寫死一種會安靜地一段都對不到——第一次
        # 跑就是這樣，107,257 筆全部「缺音檔」而沒有任何錯誤訊息。
        adir = next((p for p in (root / f"{part}-lavalier" / "lavalier",
                                 root / part / "lavalier",
                                 root / part) if p.is_dir()), None)
        if adir is None:
            print(f"找不到 {part} 的音檔目錄", file=sys.stderr)
            continue
        for spk in sorted(jdir.iterdir()):
            if not spk.is_dir():
                continue
            for jf in sorted(spk.glob("*.json")):
                try:
                    meta = json.load(open(jf, encoding="utf-8"))
                except Exception:
                    skipped["缺文字"] += 1
                    continue
                han = " ".join(str(meta.get("漢羅台文") or "").split())
                tailo = " ".join(str(meta.get("台羅數字調") or "").split())
                if not han or not tailo:
                    skipped["缺文字"] += 1
                    continue
                if has_fragment_risk(tailo):
                    skipped["台羅有碎片"] += 1
                    continue
                duration = float(meta.get("音檔長度") or 0)
                if not args.min_duration <= duration <= args.max_duration:
                    skipped["長度不合"] += 1
                    continue
                wav = adir / spk.name / f"{jf.stem}.wav"
                if not wav.is_file():
                    skipped["缺音檔"] += 1
                    continue

                # 說話人前綴要留著：naer 有 383 位，切分若以說話人分組會用到。
                name = f"{spk.name}_{jf.stem}.wav"

                # 先複製一份自己的，再從那份建硬連結。直接對原檔建硬連結會被
                # 拒絕（EPERM 而非 EXDEV）：語料屬於別的使用者，而核心的
                # protected_hardlinks 不允許連結自己沒有的檔案。複製一次約 22 GB，
                # 兩種寫法共用同一個 inode，不會再加倍。
                own = audio_root / name
                if not own.exists():
                    shutil.copyfile(wav, own)
                for variant, text in (("hanji", han), ("tailo", tailo)):
                    if variant not in variants:
                        continue
                    link = out / variant / name
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    # 硬連結而非符號連結：Path.resolve() 會解開符號連結，
                    # 兩種寫法會塌成一筆而且不會報錯。
                    link.hardlink_to(own)
                    records.append({"audio": f"{variant}/{name}", "text": text})
                kept += 1
                hours += duration / 3600

    index = out / "metadata.jsonl"
    with index.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"保留 {kept:,} 段音訊，{hours:.1f} 小時")
    print(f"寫法 {', '.join(variants)} → {len(records):,} 筆 metadata")
    for reason, count in skipped.items():
        if count:
            print(f"  略過 {reason}  {count:,}")
    print(f"\n{index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
