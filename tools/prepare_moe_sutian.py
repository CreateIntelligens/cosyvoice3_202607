#!/usr/bin/env python3
"""Shape the MOE dictionary recordings into the dual-script layout.

⚠ 授權：教育部《臺灣台語常用詞辭典》是 **CC BY-NC-ND 2.5 臺灣**（非商業性、
禁止改作）。拿來訓練幾乎必然構成改作。已去信教育部詢問，尚未收到回覆。

在取得書面同意之前，用這份資料訓練出來的模型**只能內部評估，不得公開發布、
不得商業使用**。這個限制會寫進產出目錄的 LICENCE-RESTRICTED.txt，並建議
checkpoint 一律用 `_restricted` 結尾，好讓三個月後看到檔案的人立刻知道。

同意了就把限制拿掉；被拒絕就刪除模型與這份資料。

---

資料本身：`kautian.ods` 的「詞目」與「例句」兩張表，各有一欄音檔檔名，與
`sutiau-*.zip`／`leku-*.zip` 裡的檔案 100% 對得上（2026-08-05 逐一驗證，
22,298 + 17,907 筆，兩個方向都沒有落單）。

每段音訊出兩筆，一次配漢字、一次配台羅數字調——這是 v5/v7 用媠聲驗證過的作法，
也是這裡沿用的：

    hanji/1-2-1.wav   紅嬰仔哭甲一身軀汗。
    tailo/1-2-1.wav   Ang5-enn1-a2 khau3 kah4 tsit8 sin1-khu1 kuann7.

硬連結不是符號連結：prepare_taigi_data.py 用 Path.resolve() 當鍵，而 resolve()
會解開符號連結，兩種寫法會塌成同一筆、後寫的蓋掉先寫的，兩邊文字變成一樣而且
不會報錯。詳見 tools/prepare_suisiann.py。

    python tools/prepare_moe_sutian.py --source workspace/moe_sutian \
        --out workspace/moe_prepared
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tailo import has_fragment_risk, to_numeric  # noqa: E402


RESTRICTION = """教育部《臺灣台語常用詞辭典》CC BY-NC-ND 2.5 臺灣
非商業性 · 禁止改作

用這份資料訓練出來的任何模型：
  - 只能內部評估
  - 不得公開發布
  - 不得商業使用

除非教育部書面同意。已去信詢問，截至產生本檔時尚未收到回覆。
被拒絕的話，刪除模型與這份資料。

來源 https://sutian.moe.edu.tw/zh-hant/siongkuantsuguan/
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="workspace/moe_sutian",
                        help="放 kautian.ods 與 *-wav.zip 的目錄")
    parser.add_argument("--out", default="workspace/moe_prepared")
    parser.add_argument("--audio", choices=("wav", "mp3"), default="wav")
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    args = parser.parse_args()

    import pandas as pd

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "LICENCE-RESTRICTED.txt").write_text(RESTRICTION, encoding="utf-8")

    # 詞條與例句的欄位名稱不同，音檔也在不同的壓縮檔裡。
    sheets = [
        ("詞目", "羅馬字音檔檔名", "漢字", "羅馬字", f"sutiau-{args.audio}.zip"),
        ("例句", "音檔檔名", "漢字", "羅馬字", f"leku-{args.audio}.zip"),
    ]

    audio_root = out / "_audio"
    audio_root.mkdir(exist_ok=True)
    for variant in ("hanji", "tailo"):
        (out / variant).mkdir(exist_ok=True)

    records: list[dict[str, str]] = []
    skipped = {"缺音檔": 0, "缺文字": 0, "台羅轉換失敗": 0, "長度不合": 0}
    kept = 0

    import soundfile as sf
    import torchaudio

    for sheet, namecol, hancol, romcol, archive in sheets:
        zpath = source / archive
        if not zpath.is_file():
            print(f"找不到 {zpath}，略過 {sheet}", file=sys.stderr)
            continue
        frame = pd.read_excel(source / "kautian.ods", engine="odf", sheet_name=sheet)
        frame = frame[frame[namecol].notna()]

        # 壓縮檔裡是 <分桶>/<檔名>.wav，先建索引免得逐筆掃描整個 zip。
        with zipfile.ZipFile(zpath) as zf:
            index = {Path(n).stem: n for n in zf.namelist() if not n.endswith("/")}
            # 用 to_dict 而不是 itertuples：欄位名有中文與空白，itertuples 會改名
            # 成 _1 _2 這種位置代號，對不回原本的欄位。
            for row in frame.to_dict("records"):
                stem = str(row.get(namecol) or "").strip()
                member = index.get(stem)
                if member is None:
                    skipped["缺音檔"] += 1
                    continue
                han = " ".join(str(row.get(hancol) or "").split())
                rom = str(row.get(romcol) or "")
                if not han or not rom.strip():
                    skipped["缺文字"] += 1
                    continue
                tailo = to_numeric(" ".join(rom.split()))
                # 轉不乾淨就整筆丟掉，留著只會讓模型看到位元組碎片。
                if has_fragment_risk(tailo):
                    skipped["台羅轉換失敗"] += 1
                    continue

                target = audio_root / f"{sheet}_{stem}.wav"
                if not target.exists():
                    raw = audio_root / f".raw_{stem}"
                    raw.write_bytes(zf.read(member))
                    # 原檔是 44.1 kHz 立體聲 24-bit，而 CosyVoice3 訓練用 24 kHz
                    # 單聲道。轉成模型真正會用的格式：品質不會掉（用不到的資訊
                    # 本來就會在管線裡被丟掉），檔案小 3.7 倍，而傳上叢集那條
                    # tunnel 只有 5.2 MB/s，18 GB 與 5 GB 差了快 40 分鐘。
                    try:
                        wave, sr = torchaudio.load(str(raw))
                        if wave.shape[0] > 1:
                            wave = wave.mean(0, keepdim=True)
                        if sr != 24000:
                            wave = torchaudio.transforms.Resample(sr, 24000)(wave)
                        torchaudio.save(str(target), wave, 24000,
                                        encoding="PCM_S", bits_per_sample=16)
                    except Exception:
                        skipped["缺音檔"] += 1
                        raw.unlink(missing_ok=True)
                        continue
                    finally:
                        raw.unlink(missing_ok=True)
                try:
                    info = sf.info(str(target))
                except Exception:
                    skipped["缺音檔"] += 1
                    target.unlink(missing_ok=True)
                    continue
                if not args.min_duration <= info.duration <= args.max_duration:
                    skipped["長度不合"] += 1
                    target.unlink(missing_ok=True)
                    continue

                for variant, text in (("hanji", han), ("tailo", tailo)):
                    link = out / variant / target.name
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.hardlink_to(target)
                    records.append({"audio": f"{variant}/{target.name}", "text": text})
                kept += 1

    index_path = out / "metadata.jsonl"
    with index_path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"保留 {kept:,} 段音訊 → {len(records):,} 筆 metadata（兩種寫法）")
    for reason, count in skipped.items():
        if count:
            print(f"  略過 {reason}  {count:,}")
    print(f"\n{index_path}")
    print(f"授權限制已寫入 {out / 'LICENCE-RESTRICTED.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
