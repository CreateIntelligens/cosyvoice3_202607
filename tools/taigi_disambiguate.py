#!/usr/bin/env python3
"""多義詞交給 LLM 挑讀音——但只准它「挑」，不准它「寫」。

辭典能查到「東西」在台語有 8 種講法，查不到這一句該用哪一種：

    買東西        → mih8-kiann7（物件）
    東西向的道路   → tang1-sai1（方位）

挑讀音需要上下文，那正是 LLM 會的事。但先前用翻譯 API 把整句華語改寫成台文
的做法失敗過——它會重寫句子、偏離原意。所以這裡的設計刻意不給它那個能力：

    輸入   原句 + 每個多義詞的候選清單（來自辭典）
    輸出   每個詞選第幾號
    驗證   選項不在清單裡就丟掉，退回不標音

LLM 生不出辭典以外的東西，也碰不到句子本身。它答錯最壞的情況是選了另一個
辭典裡的讀音；它答壞（幻覺、亂改）會被驗證擋下來，退回原字。**壞掉的方向是
安全的**——退回去就是現在 v7 的行為。

    python tools/taigi_disambiguate.py "明天去市場買東西。" --backend claude
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from taigi_annotate import lexicon, segment

PROMPT = """你是台語讀音選擇器。下面這句華語要交給台語 TTS 朗讀。
每個詞我已經從辭典查出候選讀音，你的工作只有一件事：依照句子的意思，
判斷每個詞在這裡該用哪一個讀音。

句子：{sentence}

{items}

規則：
- 只能從候選裡選，不可以自己寫讀音。
- 不確定就填 null，不要猜。填 null 會退回原本的漢字，那是安全的。
- 不要改動句子，不要解釋。

只輸出 JSON，格式：{{"詞": 編號或 null, ...}}"""


def candidates(text: str, max_options: int = 8):
    """挑出句子裡有多種讀音的詞。只有這些需要問 LLM。"""
    return [(w, r[:max_options]) for w, r in segment(text, lexicon())
            if r and len(r) > 1]


def ask_claude(sentence: str, items: list[tuple[str, list[str]]]) -> dict:
    listing = "\n".join(
        f"{w}：" + "  ".join(f"[{i}] {k}" for i, k in enumerate(ks))
        for w, ks in items)
    prompt = PROMPT.format(sentence=sentence, items=listing)
    out = subprocess.run(["claude", "-p", prompt], capture_output=True,
                         text=True, timeout=180).stdout
    m = re.search(r"\{.*\}", out, re.S)
    return json.loads(m.group()) if m else {}


def resolve(sentence: str, backend=ask_claude, *, verbose: bool = False):
    """回傳（標好的句子, 決策紀錄）。LLM 給的每個選擇都驗證過才採用。"""
    items = candidates(sentence)
    if not items:
        return sentence, []

    try:
        picked = backend(sentence, items)
    except Exception as e:                      # 服務掛掉不該讓 TTS 跟著掛
        print(f"  LLM 呼叫失敗（{type(e).__name__}），全部退回原字", file=sys.stderr)
        picked = {}

    chosen, log = {}, []
    for word, ks in items:
        idx = picked.get(word)
        # 驗證：必須是整數、必須落在這個詞自己的候選範圍內
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(ks):
            log.append((word, None, "退回原字" if idx is None else f"無效選擇 {idx!r}，退回原字"))
            continue
        chosen[word] = ks[idx]
        log.append((word, ks[idx], f"從 {len(ks)} 個候選中選 [{idx}]"))

    pieces = []
    for word, readings in segment(sentence, lexicon()):
        if readings and word in chosen:
            pieces.append(f" {chosen[word]} ")
        elif readings and len(readings) == 1:
            pieces.append(f" {readings[0]} ")     # 唯一讀音不必問
        else:
            pieces.append(word)
    text = re.sub(r" {2,}", " ", "".join(pieces))
    text = re.sub(r"\s+([，。！？；：、])", r"\1", text).strip()
    return text, log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--show", action="store_true", help="列出每個詞怎麼決定的")
    a = ap.parse_args()

    out, log = resolve(a.text)
    print(out)
    if a.show:
        for word, kip, why in log:
            print(f"  {word:<6} → {kip or '（不標）':<16} {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
