#!/usr/bin/env python3
"""量「挑讀音」挑得準不準——不必聽，也不必人工標答案。

答案卷是現成的：naer 的 107,292 句同時有華語與台語漢字兩種寫法。

    text        接著來看新竹市明天二十六號的天氣
    text_taigi  紲落來看新竹市明仔載二十六號的天氣

「接著→紲落來」「明天→明仔載」正是我們在追的華語詞≠台語詞。所以一個詞的
候選讀音裡，哪一個的台語漢字出現在 text_taigi 裡，哪一個就是這句的答案。

    python tools/taigi_eval_disambiguation.py --limit 20000
    → 按辭典詞頻挑 57.3%（57,313 個決策）

這是基準線。任何消歧做法都要贏過它才有意義。要評自己的做法就傳 resolver：

    from taigi_eval_disambiguation import evaluate
    evaluate(pick=my_resolver, limit=2000)

答案卷本身有雜訊：用的是子字串比對，「知影」與「知」都算命中「知」，而
「佮」的 kap/kah 是兩種腔口寫法卻算不同答案。所以絕對數字偏低，**拿它比較
不同做法是有效的，拿它當真實準確率會低估**。
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taigi_annotate import add_tone_digits, lexicon, segment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAER = Path("/mnt/nas/cosyvoice_taipei_1_no5/naer_mandarin_20260803.jsonl")
CHHOETAIGI = ROOT / "workspace/chhoetaigi/ChhoeTaigi_TaihoaSoanntengTuichiautian.csv"


def kip_to_hanji() -> dict[str, set[str]]:
    """台羅 → 台語漢字。用來把候選對回答案卷裡的寫法。"""
    import pandas as pd
    ct = pd.read_csv(CHHOETAIGI).dropna(subset=["KipInput", "HanLoTaibunKip"])
    out = collections.defaultdict(set)
    for k, h in zip(ct.KipInput, ct.HanLoTaibunKip):
        out[add_tone_digits(str(k).strip())].add(str(h).strip())
    return out


def items(limit: int):
    """產出（句子, 詞, 候選, 答案）——只收答案唯一的，避免歧義污染評測。"""
    lex, k2h = lexicon(), kip_to_hanji()
    with NAER.open(encoding="utf-8") as f:
        for line in itertools.islice(f, limit):
            r = json.loads(line)
            hoa, tai = r["text"], r["text_taigi"]
            for word, cands in segment(hoa, lex):
                if not cands or len(cands) < 2:
                    continue
                gold = [c for c in cands if any(h and h in tai for h in k2h.get(c, ()))]
                if len(gold) == 1:
                    yield hoa, word, cands[:8], gold[0]


def evaluate(pick=None, limit: int = 20000):
    """pick(word, cands, sentence) -> 讀音。給 None 就評辭典詞頻這條基準線。"""
    right = total = 0
    wrong = []
    for sent, word, cands, gold in items(limit):
        total += 1
        got = cands[0] if pick is None else pick(word, cands, sent)
        if got == gold:
            right += 1
        elif len(wrong) < 20:
            wrong.append((word, got, gold, sent[:30]))
    return right, total, wrong


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=20000, help="讀幾句 naer")
    ap.add_argument("--quiz", type=int, metavar="N",
                    help="出 N 題考卷（答案另存），拿去問 LLM 或人")
    a = ap.parse_args()

    if a.quiz:
        import random
        k2h = kip_to_hanji()
        pool = list(items(a.limit))
        random.seed(7)
        random.shuffle(pool)
        for i, (sent, word, cands, gold) in enumerate(pool[:a.quiz]):
            opts = "  ".join(f"[{j}]{k}({'/'.join(sorted(k2h.get(k, {'?'}))[:1])})"
                             for j, k in enumerate(cands))
            print(f"{i}\t{sent}\t詞={word}\t{opts}")
        json.dump([g for *_, g in pool[:a.quiz]], open("quiz_gold.json", "w"),
                  ensure_ascii=False)
        print(f"答案存到 quiz_gold.json（{a.quiz} 題）", file=sys.stderr)
        return 0

    right, total, wrong = evaluate(limit=a.limit)
    print(f"  基準線（辭典詞頻）  {right:,}／{total:,} = {right/total*100:.1f}%")
    print("\n  挑錯的例子：", file=sys.stderr)
    for w, got, want, s in wrong[:10]:
        print(f"    {w:<4} {got:<16} → {want:<16} 「{s}」", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
