#!/usr/bin/env python3
"""把華語輸入裡「模型會念錯的詞」換成台羅標音，其餘原字不動。

v7 讀得懂漢字混台羅的混排（零樣本成立），而標音會壓過模型自己的偏好——
「乾坤」不標音 3 個 seed 對 2 個，標音 5/5。這支工具是把那件事自動化。

它處理兩種錯法，而這兩種在模型端是同一個修法：

    破音字        乾坤 該念 khian5-khun1，模型念成 kan1-khun1
    華語詞≠台語詞  東西 該念 mih8-kiann7（物件），模型逐字念成 tong1-se1

第二種先前被歸類為「標音救不了」，因為它是詞彙不是讀音。但辭典查得到
東西→mih8-kiann7，而標音送進模型的是讀音不是字——所以救得了。

    python tools/taigi_annotate.py "明天去市場買東西。"
    → 明天去市場買 mih8-kiann7。

辭典兩本，都在 workspace/：
    ChhoeTaigi 台華線頂對照典  91,339 筆   華語詞 → 台羅（本工具的主力）
    教育部臺灣台語常用詞辭典    29,591 筆   台語漢字 → 台羅（補漏、當真值）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHHOETAIGI = ROOT / "workspace/chhoetaigi/ChhoeTaigi_TaihoaSoanntengTuichiautian.csv"
KAUTIAN = ROOT / "workspace/moe_sutian/kautian.ods"
CACHE = ROOT / "workspace/chhoetaigi/lexicon.json"

# 台羅音節結尾是 p/t/k/h 的是入聲，沒標數字就是第 4 調；其餘沒標就是第 1 調。
STOPS = ("p", "t", "k", "h")
MAX_WORD = 6          # 最長比對幾個字。辭典裡 6 字以上的詞不到 0.3%
HAN = re.compile(r"[一-鿿㐀-䶿]")


def add_tone_digits(kip: str) -> str:
    """ChhoeTaigi 省略第 1、4 調的數字；v7 的訓練資料每個音節都有數字。

    不補的話模型看到的是訓練時沒出現過的格式。這是會安靜出錯的那種細節——
    模型照樣念得出東西，只是念的不是你以為的那個。
    """
    out = []
    for piece in re.split(r"([^A-Za-z0-9]+)", kip):
        if not piece or not piece[0].isalpha():
            out.append(piece)
            continue
        if piece[-1].isdigit():
            out.append(piece)
        elif piece[-1].lower() in STOPS:
            out.append(piece + "4")
        else:
            out.append(piece + "1")
    return "".join(out)


def _looks_clean(kip: str) -> bool:
    """只收乾淨的台羅。辭典裡混著解釋文字、替代寫法與空條目。"""
    return bool(kip) and re.fullmatch(r"[A-Za-z0-9 \-]+", kip) is not None


def build_lexicon() -> dict[str, list[str]]:
    """華語詞 → 台羅讀音（可能多個，依辭典出現次數排序）。"""
    import pandas as pd

    counts: dict[str, dict[str, int]] = {}
    ct = pd.read_csv(CHHOETAIGI).dropna(subset=["HoaBun", "KipInput"])
    for hoa, kip in zip(ct.HoaBun, ct.KipInput):
        hoa = str(hoa).strip()
        kip = add_tone_digits(str(kip).strip())
        # 華語欄常寫成「不然；還是沒有」這種帶解釋的形式，只收純詞
        if not hoa or not HAN.search(hoa) or len(hoa) > MAX_WORD:
            continue
        if re.search(r"[，。；、（）()\s]", hoa) or not _looks_clean(kip):
            continue
        counts.setdefault(hoa, {})
        counts[hoa][kip] = counts[hoa].get(kip, 0) + 1

    # 同一個詞出現愈多次，代表愈是主流講法——多義時當作預設
    return {w: [k for k, _ in sorted(v.items(), key=lambda kv: -kv[1])]
            for w, v in counts.items()}


@lru_cache(maxsize=1)
def lexicon() -> dict[str, list[str]]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    lex = build_lexicon()
    CACHE.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    return lex


def segment(text: str, lex: dict[str, list[str]]):
    """最長詞優先切分。

    逐字查是行不通的：「乾坤」拆開查會得到 kan1（乾燥的乾）+ khun1，
    而整個詞查是 khian5-khun1。破音字要的是詞層級的知識。
    """
    out, i = [], 0
    while i < len(text):
        for n in range(min(MAX_WORD, len(text) - i), 0, -1):
            w = text[i:i + n]
            if w in lex:
                out.append((w, lex[w]))
                i += n
                break
        else:
            out.append((text[i], None))
            i += 1
    return out


def annotate(text: str, *, only_unambiguous: bool = True,
             wordlist: set[str] | None = None, mark: str = "replace"):
    """回傳（標好的文字, 改了哪些詞）。

    only_unambiguous  辭典給出多種讀音時不標。挑錯讀音比不標更糟——
                      不標至少還有模型自己的判斷，標錯是直接指定念錯。
    wordlist          只標這些詞。給 None 表示標所有查得到的
    mark              replace = 原字換成台羅（訓練時見過的形式）
                      append  = 原字保留，台羅放在括號裡
    """
    lex = lexicon()
    pieces, changed = [], []
    for word, readings in segment(text, lex):
        if readings is None:
            pieces.append(word)
            continue
        if wordlist is not None and word not in wordlist:
            pieces.append(word)
            continue
        if only_unambiguous and len(readings) > 1:
            pieces.append(word)
            continue
        kip = readings[0]
        pieces.append(f" {kip} " if mark == "replace" else f"{word}({kip})")
        changed.append((word, kip, len(readings)))
    # 換字會在中文之間留下多餘空白，收乾淨
    return re.sub(r"\s+([，。！？；：、])", r"\1", "".join(pieces)).strip(), changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="要標音的華語文字；省略則讀 stdin")
    ap.add_argument("--all", action="store_true",
                    help="多義詞也標（取最常見的讀音）。預設只標唯一讀音的詞")
    ap.add_argument("--words", help="只標這些詞，逗號分隔")
    ap.add_argument("--append", action="store_true", help="保留原字，台羅放括號裡")
    ap.add_argument("--show", action="store_true", help="列出改了哪些詞")
    a = ap.parse_args()

    text = a.text if a.text else sys.stdin.read().strip()
    out, changed = annotate(
        text,
        only_unambiguous=not a.all,
        wordlist=set(a.words.split(",")) if a.words else None,
        mark="append" if a.append else "replace",
    )
    print(out)
    if a.show:
        for w, k, n in changed:
            note = "" if n == 1 else f"  ← 辭典有 {n} 種讀音，取最常見的"
            print(f"  {w} → {k}{note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
