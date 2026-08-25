#!/usr/bin/env python3
"""華語 → 台語漢字。不轉羅馬字，只換詞。

台羅那條路的問題是語調：v7 只用 4.62 小時學台羅，唸起來像把詞硬黏在一起。
而**台語漢字是它的訓練分布本身**——v6 用 naer 290 小時訓練，naer 的原文就是
台語漢字（風颱、物件、海豬）。所以餵台語漢字，語調跟餵華語一樣自然，詞卻是對的。

    python tools/taigi_hanji.py "颱風白海豚鄰近，氣象署發布海上颱風警報。"
    → 風颱白海豬近兜，氣象署發佈海上風颱警報。

對照表 23,528 個詞，選字規則三層：

    1. 只留教育部推薦用字      「懶」擋掉，剩「咱」
    2. 語料票數×3 + 辭典票數    語料是「人真的這樣寫」，辭典是「可以這樣寫」
    3. 同分取字數少的

查不到的詞原字留著。**這裡混不混沒關係**——輸出全是漢字，不會出現台羅夾漢字
那種 1/10 會幻覺的形態。這是這條路比標音路好的第二個地方。
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taigi_annotate import MAX_WORD, lexicon, segment  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "workspace/chhoetaigi/hanji_table.json"


def table(version: str | None = None) -> dict[str, str]:
    """三層合起來：人工修正 > LLM 補的 > 語料對齊。見 taigi_store。

    底表是從 naer 的 107,231 句雙寫法統計對齊出來的：兩邊都斷詞，算 Dice 共現
    強度，門檻是華語詞出現 ≥10 次、共現 ≥5 次、Dice ≥0.35。

    先前的做法是「拿辭典候選去台語句子裡找有沒有出現」，那不是對齊，而是所有
    錯法的來源——「物」是「物件」的子字串所以短的假贏；台華線頂的「走找」拿華語
    字充數卻擠掉了正確的「揣」；單字「張」只收量詞義所以人名變成「條飛」。
    對齊把這些一次解決，而且單字不必特別處理（「張」自己就對齊到不換）。

    門檻是掃出來的：再鬆一級，罕見詞的假答案就會混進來（「走調→刜」
    「廢銅爛鐵→簿仔紙」——樣本少時 Dice 會虛高）。代價是覆蓋率只有 1,427 詞。

    不快取——人工修正與 LLM 補的隨時會變，而查表很便宜。
    """
    from taigi_store import merged
    return {w: h for w, h in merged(version).items() if h != w}


def convert(text: str, *, log_unknown: bool = False, version: str | None = None,
            extra: dict[str, str] | None = None):
    """回傳（台語漢字, 換了哪些詞, 查不到的詞）。

    查不到就留原字——那是現在 none 模式的行為，也就是最壞情況不會更差。
    查不到的詞可以記下來，之後離線批次問 LLM 補進表裡，線上永遠不等 LLM。
    """
    tbl, lex = dict(table(version)), lexicon()
    if extra:
        tbl.update(extra)          # 這一次請求專用，不寫進任何一層
    out, changed, unknown = [], [], []
    for word, readings in segment(text, lex):
        got = tbl.get(word)
        if got:
            out.append(got)
            changed.append((word, got))
        else:
            out.append(word)
            # 只有「像詞的東西」才值得記，標點與單一標點符號不算
            if len(word) >= 2 and word not in tbl:
                unknown.append(word)
    if log_unknown and unknown:
        from taigi_store import log_unknown as _log
        _log(text, unknown)      # 寫不進去會回 False，不擋住合成
    return "".join(out), changed, unknown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?")
    ap.add_argument("--show", action="store_true", help="列出換了哪些詞")
    a = ap.parse_args()
    text = a.text if a.text else sys.stdin.read().strip()
    out, changed, unknown = convert(text)
    print(out)
    if a.show:
        for w, h in changed:
            print(f"  {w} → {h}", file=sys.stderr)
        if unknown:
            print(f"  查不到（原字留著）：{'、'.join(unknown)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
