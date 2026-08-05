#!/usr/bin/env python3
"""Convert Tâi-lô romanisation from diacritics to numeric tone notation.

CosyVoice3 reads text through Qwen's tokenizer, and Tâi-lô written with
diacritics does not survive it.  The eighth-tone mark is a combining vertical
line (U+030D) with no precomposed form, so every syllable carrying it falls
back to raw UTF-8 bytes:

    tsi̍t ji̍t la̍t si̍p   →  ts | i | � | � | t | ji | � | � | t | …   8 fragments
    tsit8 jit8 lat8 sip8  →  ts | it | 8 | jit | 8 | lat | 8 | sip | 8   none

Numeric notation costs nothing — it is the same romanisation, and the MOE
dictionary and TAT-TTS both publish it — so annotating text for the model means
converting to numbers first.

Tone is recovered from the mark on the syllable's vowel:

    (none)  1 or 4   depends on the coda: a stop ending is 4, anything else 1
    ́       2        acute
    ̀       3        grave
    ̂       5        circumflex
    ̌       6        caron — merged into 2 in most of Taiwan, kept distinct here
    ̄       7        macron
    ̍       8        vertical line above
    ̋       9        double acute — only in a few loanwords

    python tools/tailo.py --check          # 用教育部辭典驗證
    python tools/tailo.py "tsi̍t ji̍t"      # 轉換一段文字
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata


# 附加符號 → 調號。用組合字元（NFD 之後的形式）當鍵，因為台羅的第 8 調
# （U+030D）根本沒有預組字，只能以組合字元處理，其他調也就一併統一。
TONE_MARKS = {
    "́": 2,   # ́  acute
    "̀": 3,   # ̀  grave
    "̂": 5,   # ̂  circumflex
    "̌": 6,   # ̌  caron
    "̄": 7,   # ̄  macron
    "̍": 8,   # ̍  vertical line above
    "̋": 9,   # ̋  double acute
}

# 入聲韻尾。沒有調號時，收 -p -t -k -h 的是第 4 調，其餘是第 1 調。
STOPS = ("p", "t", "k", "h")

# 音節內部允許的字母。ⁿ 是白話字的鼻化符號，台羅寫成 nn，兩種都收，
# 因為語料常常混用。
LETTERS = re.compile(r"^[a-zA-Zⁿ]+$")

# 排版慣例：i 和 j 上面要加調號時，有些語料改用無點的 ı／ȷ，才不會和原本的
# 點擠在一起。媠聲就這樣寫（tsı̍t 而非 tsi̍t），數量不多但足以讓整句轉換失敗。
DOTLESS = {"ı": "i", "ȷ": "j", "I": "I", "J": "J"}


def syllable_to_numeric(syllable: str) -> str | None:
    """Return the syllable with its diacritic replaced by a tone number.

    Returns None when the input is not a Tâi-lô syllable at all — punctuation,
    a Han character, a bare digit — so callers can pass text through unchanged
    rather than mangling it.
    """
    decomposed = unicodedata.normalize("NFD", syllable)
    tone = 0
    letters = []
    for ch in decomposed:
        if ch in TONE_MARKS:
            # 同一個符號重複是來源資料的錯字（媠聲有一筆 m̄̄-thang），意思沒有
            # 歧義，照單一符號處理。兩個「不同」的調號才是真的無法判讀。
            if tone and tone != TONE_MARKS[ch]:
                return None
            tone = TONE_MARKS[ch]
        elif unicodedata.category(ch) == "Mn":
            continue                 # 其他組合字元（不該出現）直接丟掉
        else:
            letters.append(DOTLESS.get(ch, ch))

    base = "".join(letters)
    if not base or not LETTERS.match(base):
        return None
    base = base.replace("ⁿ", "nn")

    if not tone:
        tone = 4 if base[-1].lower() in STOPS else 1
    return f"{base}{tone}"


# 音節之間用連字號或空白分隔；標點、漢字、既有的數字都原樣保留。
#
# 組合字元（U+0300–U+036F）要明確寫進來。Python 的 \w 不含 Mn 類別，所以
# 少了這一段，「tsi̍t」會在第 8 調的符號處斷成「tsi」和「t」，各自被判成
# 第 1 調與第 4 調，變成 tsi1̍t4——符號還留著，音節也錯了。
_TOKEN = re.compile(r"(?:[^\W\d_]|[̀-ͯ])+", re.UNICODE)


def to_numeric(text: str) -> str:
    """Convert every Tâi-lô syllable in `text`, leaving everything else alone."""
    def swap(match: re.Match[str]) -> str:
        got = syllable_to_numeric(match.group())
        return got if got is not None else match.group()
    return _TOKEN.sub(swap, text)


def has_fragment_risk(text: str) -> bool:
    """True when `text` still carries a combining mark the tokenizer will break."""
    return any(ch in TONE_MARKS for ch in unicodedata.normalize("NFD", text))


def check(path: str) -> int:
    """Convert every romanisation in the MOE dictionary and report what failed."""
    import pandas as pd

    total = converted = 0
    leftover: list[str] = []
    for sheet in ("詞目", "例句"):
        frame = pd.read_excel(path, engine="odf", sheet_name=sheet)
        for value in frame["羅馬字"].dropna().astype(str):
            # 【文】【白】是文白讀標記，斜線是又讀，兩者都不是音節的一部分
            cleaned = re.sub(r"【[^】]*】", " ", value)
            out = to_numeric(cleaned)
            total += 1
            if not has_fragment_risk(out):
                converted += 1
            else:
                if len(leftover) < 10:
                    leftover.append(f"{value}  →  {out}")

    print(f"教育部辭典 {total:,} 筆羅馬字")
    print(f"  轉換後無碎片風險  {converted:,}  ({converted / total * 100:.2f}%)")
    if leftover:
        print(f"  仍有附加符號 {total - converted:,} 筆，前幾筆：")
        for line in leftover:
            print(f"    {line}")
    return 0 if converted == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="*", help="要轉換的文字；省略則從 stdin 讀")
    parser.add_argument("--check", metavar="ODS", nargs="?",
                        const="workspace/moe_sutian/kautian.ods",
                        help="拿教育部辭典的羅馬字整批驗證")
    args = parser.parse_args()

    if args.check:
        return check(args.check)

    source = " ".join(args.text) if args.text else sys.stdin.read()
    sys.stdout.write(to_numeric(source))
    if args.text:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
