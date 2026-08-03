#!/usr/bin/env python3
"""Rewrite the NAER transcripts from Taiwanese orthography into Mandarin.

NAER pairs clean read speech with Taiwanese text (台文漢字 plus 台羅 romanisation):
「毋過」, 「佇」, 「m̄-koh...lóng講bē清楚」.  The model we are training is asked at
inference time to read *Mandarin* input, so those 94k utterances teach a task
nobody performs — yet the ablation showed the naer-only model was the best of the
six, because Taiwanese and Mandarin share most content-word characters and the
read-speech alignment is tight.

Converting the text while keeping the audio turns that corpus into 214 hours of
genuine 「華語 → 台語音」 material.  The audio paths are untouched, so on the
cluster only the Parquet stage has to be rebuilt: the embeddings and speech
tokens stay valid.

Only unique sentences are sent to the API — 383 speakers read the same scripts,
so 107,292 utterances collapse to 34,488 distinct lines.

    export GEMINI_API_KEY=...
    python tools/naer_taigi_to_mandarin.py --limit 200 --out pilot.jsonl   # 先試水溫
    python tools/naer_taigi_to_mandarin.py --out naer_mandarin.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import requests


API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = """你是台語（臺灣閩南語）與華語的轉換專家。

下面每一行是一句台語文字，可能包含：
- 台文漢字（例如「毋過」「佇」「袂」「攏」「欲」「較」「猶」「紲落來」）
- 台羅羅馬字（例如「m̄-koh」「ê」「lóng」「bē」「pha」）
- 兩者混寫

請把每一行改寫成**意思完全相同的華語（標準中文）書面文字**。

規則：
1. 一行對一行，輸出的行數必須和輸入完全相同。
2. 每行格式為「編號<TAB>華語」，編號沿用輸入的編號。
3. 只轉換文字，不要增減內容、不要解釋、不要加註音。
4. 數字保持原樣（「二十六」不要改成「26」）。
5. 標點符號沿用原文。
6. 羅馬字要一併轉成對應的華語詞（m̄-koh→不過、ê→的、lóng→都、bē→不會、pha→趴）。
7. 專有名詞、地名、人名保持原樣。
8. 若某行本來就是華語，原樣輸出。

輸入：
{lines}

輸出："""

ROMAN = re.compile(r"[a-zA-Zāáàâǎēéèêěīíìîǐōóòôǒūúùûǔńňǹḿ̄́̀̂̌]")

# 只列「絕不會出現在標準中文裡」的字。刻意排除「怎」「較」——「怎麼」「比較」
# 是標準中文，把它們當成殘留會產生大量假警報。
TAIGI_ONLY = re.compile(
    r"[毋佇袂攏猶紲遮啥咧个囡濟捌拄敆賰媠迵఍఍఍఍఍఍఍఍఍఍഍഍഍഍഍഍഍഍഍഍഍഍഍഍഍഍഍഍]"
    r"|[歹]勢|有影|無閒|交關|十外冬|按呢|甲若|創啥|тр"
)


def looks_untranslated(taigi: str, mandarin: str) -> str:
    """回傳可疑的原因，空字串代表看起來沒問題。"""
    if not mandarin.strip():
        return "空白"
    if ROMAN.search(mandarin) and not ROMAN.search(re.sub(r"[A-Za-z]{2,}", "", mandarin)):
        pass  # 保留原樣的英文縮寫（PTCG 之類）不算殘留
    elif ROMAN.search(mandarin):
        return "仍含羅馬字"
    if TAIGI_ONLY.search(mandarin):
        return "仍含台文專用詞"
    ratio = len(mandarin) / max(len(taigi), 1)
    # 音檔沒變，文字長度大幅偏離就代表增刪了內容
    if ratio < 0.45 or ratio > 1.8:
        return f"長度比 {ratio:.2f} 異常"
    return ""


def load_unique(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Return the distinct transcripts and, for each, the utterances using it."""
    by_text: dict[str, list[str]] = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        by_text.setdefault(record["text"].strip(), []).append(record["utterance_id"])
    return list(by_text), by_text


def convert(batch: list[tuple[int, str]], model: str, key: str,
            timeout: int, retries: int) -> dict[int, str] | None:
    lines = "\n".join(f"{index}\t{text}" for index, text in batch)
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(lines=lines)}]}],
        # Deterministic: this is a transcription task, not a creative one, and a
        # rerun after a crash should reproduce the same file.
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 65536},
    }
    for attempt in range(retries):
        try:
            response = requests.post(
                API.format(model=model), params={"key": key},
                json=body, timeout=timeout,
            )
        except requests.RequestException as error:
            print(f"      連線失敗（第 {attempt + 1} 次）: {error}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
            continue
        if response.status_code == 429:
            wait = 20 * (attempt + 1)
            print(f"      額度受限，等 {wait} 秒", file=sys.stderr)
            time.sleep(wait)
            continue
        if response.status_code != 200:
            print(f"      HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
            continue

        payload = response.json()
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            print(f"      回應格式異常: {str(payload)[:200]}", file=sys.stderr)
            time.sleep(2)
            continue

        out: dict[int, str] = {}
        for row in text.splitlines():
            if "\t" not in row:
                continue
            head, _, value = row.partition("\t")
            head = head.strip()
            if head.isdigit() and value.strip():
                out[int(head)] = value.strip()
        # A short reply means dropped lines; retrying is cheaper than silently
        # shipping a mapping with holes in it.
        if len(out) >= len(batch) * 0.9:
            return out
        print(f"      只回了 {len(out)}/{len(batch)} 行，重試", file=sys.stderr)
        time.sleep(2)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", default="/mnt/nas/dataset202607_1/naer/metadata.jsonl")
    parser.add_argument("--out", required=True, help="輸出的對照表 jsonl")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="只轉前 N 句（試水溫用）")
    parser.add_argument("--sample", action="store_true", help="配合 --limit：隨機抽樣而非取前 N 句")
    parser.add_argument("--seed", type=int, default=1986)
    parser.add_argument("--sleep", type=float, default=1.0, help="每批之間的間隔秒數")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--verify", action="store_true",
                        help="檢查既有結果，把可疑的挑出來重轉")
    parser.add_argument("--emit-metadata", default="",
                        help="依對照表產生新的 metadata.jsonl（audio 路徑不變，只換 text）")
    args = parser.parse_args()

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        print("請先設定 GEMINI_API_KEY", file=sys.stderr)
        return 2

    source = Path(args.metadata)
    if not source.is_file():
        print(f"找不到 {source}", file=sys.stderr)
        return 2

    if args.emit_metadata:
        mapping = {}
        for line in Path(args.out).open(encoding="utf-8"):
            record = json.loads(line)
            mapping[record["taigi"]] = record["mandarin"]
        dest = Path(args.emit_metadata)
        dest.parent.mkdir(parents=True, exist_ok=True)
        kept = missing = 0
        with dest.open("w", encoding="utf-8") as sink:
            for line in source.open(encoding="utf-8"):
                record = json.loads(line)
                mandarin = mapping.get(record["text"].strip())
                if mandarin is None:
                    missing += 1
                    continue   # 沒轉到的整筆略過，寧可少也不要混入台文
                record["text_taigi"] = record["text"]   # 留著原文方便日後比對
                record["text"] = mandarin
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
        print(f"{dest}\n  寫出 {kept:,} 筆，略過 {missing:,} 筆（對照表沒有）")
        return 0

    texts, by_text = load_unique(source)
    total_utts = sum(len(v) for v in by_text.values())
    print(f"{source}\n  {total_utts:,} 句錄音  →  {len(texts):,} 句唯一文字")

    if args.limit:
        if args.sample:
            # 刻意混入含羅馬字的句子，那是最容易轉錯的一類
            random.Random(args.seed).shuffle(texts)
            roman = [t for t in texts if ROMAN.search(t)][: args.limit // 3]
            plain = [t for t in texts if not ROMAN.search(t)][: args.limit - len(roman)]
            texts = roman + plain
            random.Random(args.seed).shuffle(texts)
        else:
            texts = texts[: args.limit]
        print(f"  只處理 {len(texts):,} 句"
              f"（含羅馬字 {sum(1 for t in texts if ROMAN.search(t))} 句）")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 續傳：已經轉過的就不重送，中斷後直接重跑同一行指令即可
    done: dict[str, str] = {}
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            record = json.loads(line)
            done[record["taigi"]] = record["mandarin"]
        print(f"  已有 {len(done):,} 句轉換結果，略過")

    if args.verify:
        suspects = {t: looks_untranslated(t, m) for t, m in done.items()}
        suspects = {t: why for t, why in suspects.items() if why}
        if not suspects:
            print("  全部通過檢查")
            return 0
        from collections import Counter
        print(f"  {len(suspects):,} / {len(done):,} 句可疑 "
              f"({len(suspects)/len(done):.1%})")
        for why, n in Counter(suspects.values()).most_common():
            print(f"    {why}: {n}")
        # 重轉時把原譯文一併給模型，讓它知道上一次哪裡沒轉乾淨
        todo = list(suspects)
        for t in todo:
            done.pop(t, None)
        rewritten = [json.dumps({"taigi": t, "mandarin": m}, ensure_ascii=False)
                     for t, m in done.items()]
        out_path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")
    else:
        todo = [t for t in texts if t not in done]
    if not todo:
        print("  沒有需要轉換的句子")
        return 0

    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    print(f"  待轉 {len(todo):,} 句，分成 {len(batches)} 批\n")

    started, failed = time.time(), 0
    with out_path.open("a", encoding="utf-8") as sink:
        for number, batch in enumerate(batches, start=1):
            indexed = list(enumerate(batch))
            result = convert(indexed, args.model, key, args.timeout, args.retries)
            if result is None:
                failed += len(batch)
                print(f"  [{number}/{len(batches)}] ✗ 整批失敗", file=sys.stderr)
                continue
            for index, taigi in indexed:
                mandarin = result.get(index)
                if mandarin:
                    sink.write(json.dumps({"taigi": taigi, "mandarin": mandarin}, ensure_ascii=False) + "\n")
                else:
                    failed += 1
            sink.flush()
            rate = number / max(time.time() - started, 1e-9)
            left = (len(batches) - number) / max(rate, 1e-9)
            print(f"  [{number}/{len(batches)}] ✓  剩餘約 {left / 60:.1f} 分鐘")
            time.sleep(args.sleep)

    print(f"\n完成，失敗 {failed} 句 → {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
