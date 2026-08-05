#!/usr/bin/env python3
"""Build a Traditional→Simplified map limited to characters the tokenizer breaks.

CosyVoice3 reads text through Qwen's tokenizer, whose vocabulary is
Simplified-leaning: a Traditional character with no shared glyph falls back to
raw UTF-8 bytes.  20.6% of the distinct characters in our training text end up
that way — 嬤, 颱, 諒, 蹤, 勸 — so the model never sees them as characters at
all, only as meaningless byte fragments.  Feeding the Simplified form instead
produces the right Taiwanese pronunciation for words that were consistently
wrong before (verified on 三顧茅廬 across three seeds).

Traditional→Simplified is many-to-one, and two earlier versions of this refused
mappings on that basis — first any collision at all, then the subset that seemed
to collapse different readings.  Listening tests contradicted both, in the same
direction each time:

  兇/颱  wrong untouched, correct as 凶/台 — they are homophones, so nothing is lost
  乾杯   read as 前杯 untouched, correct as 干杯 — 乾 is a polyphone and the merge
         leaves the model one reading instead of two

Merging resolves ambiguity rather than creating it, so nothing is excluded now
beyond the technical case where the Simplified form fragments too.

A mapping is kept when both hold:

  1. the Traditional character byte-fragments
  2. its Simplified form is a single, complete token

The unrestricted table is written alongside it, because converting characters
that tokenise fine turns out to help as well (乾杯 is the example) — which of the
two to use is a listening question, so both are available.  Both are JSON, so the
runtime reads a table it can print instead of depending on a conversion library.

    python tools/build_token_safe_map.py --tokenizer <path> --out tools/data/t2s_safe.json
"""

from __future__ import annotations

import argparse

import json
from pathlib import Path
import sys


# 常用漢字大致落在這個區間；擴充區的字在語音語料裡幾乎不會出現，
# 逐一檢查兩萬字已經足夠涵蓋，也讓產出的表小到可以人工翻閱。
CJK_START, CJK_END = 0x4E00, 0x9FFF


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="CosyVoice3 用的 tokenizer 目錄")
    parser.add_argument("--out", default="tools/data/t2s_safe.json")
    parser.add_argument("--report", default="", help="另外寫一份排除清單，說明每個字為何被排除")
    args = parser.parse_args()

    from opencc import OpenCC
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    # tw2s 而非 t2s：台灣把 zhe/zháo/zhuó 都寫成「著」，大陸用「着」，而 t2s 的
    # 字表沒有這條規則。tw2s 有，且處理得精準——看著→看着、著急→着急，但
    # 著作（zhù）保留。再往上的 tw2sp 才會換詞彙（滑鼠→鼠标），那不是我們要的。
    cc = OpenCC("tw2s")

    def fragments(text: str) -> bool:
        """Qwen 找不到這個字時會退化成 UTF-8 位元組，解碼就出現替代字元。"""
        return any("�" in tok.decode([i]) for i in tok.encode(text, add_special_tokens=False))

    chars = [chr(c) for c in range(CJK_START, CJK_END + 1)]
    simplified = {c: cc.convert(c) for c in chars}


    # 這裡本來擋著一份「合併會造成同形異音」的清單（麵/面、後/后、乾/干…），
    # 出發點是怕合併抹掉區別。兩次聽測把這個顧慮推翻了，而且方向相反：
    #
    #   「兇」「颱」不轉會念錯，轉成「凶」「台」就對——它們本來就同音，合併無損。
    #   「乾杯」不轉會念成「前杯」，轉成「干杯」就對——「乾」是破音字，
    #   合併反而給了模型唯一解。
    #
    # 合併消除歧義，不是製造歧義。所以不再排除任何字，只保留「轉了也沒用」
    # （簡體仍碎片）這個技術性條件。
    safe: dict[str, str] = {}
    skipped: dict[str, list[str]] = {"沒有碎片": [], "簡體仍碎片": []}
    for trad in chars:
        simp = simplified[trad]
        if simp == trad or not fragments(trad):
            skipped["沒有碎片"].append(trad)
            continue
        if fragments(simp):
            skipped["簡體仍碎片"].append(trad)
            continue
        safe[trad] = simp

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(safe, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")

    # 完整表一併產生，這樣執行端不必安裝 opencc——它只需要讀 JSON。
    full = {t: s for t, s in simplified.items() if s != t}
    full_path = out.with_name(out.stem.replace("_safe", "_full") + out.suffix)
    full_path.write_text(json.dumps(full, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")

    # 逐字轉換會把「乾坤」變成「干坤」——簡化並沒有動 qián 這個讀音，
    # 是 OpenCC 的詞組表在保護它（規則長這樣：乾坤→乾坤，恆等但優先套用）。
    # 少了這一步，破音字就會被轉錯，所以把詞組表一併帶出來。
    # 注意這裡取的是 TSPhrases（破音字消歧），不是 tw2sp 的詞彙替換——
    # 後者會把「滑鼠」變成「鼠标」，那不是我們要的。
    import opencc

    # tw2s 疊了兩層詞組表：TWPhrases 處理台灣用字，TSPhrases 保護破音字
    dictionary = Path(opencc.__file__).with_name("dictionary")
    src = dictionary / "TSPhrases.txt"
    if not src.is_file():
        print(f"找不到 {src}，詞組規則未匯出——破音字（乾坤、乾隆）會被轉錯", file=sys.stderr)
        return 0

    # tw2s 的 conversion_chain 有兩段，兩段都靠詞組表保護破音字：
    #   TWVariantsRevPhrases  著作→著作、著名→著名   （保住 zhù，其餘才轉「着」）
    #   TSPhrases             乾坤→乾坤、乾隆→乾隆   （保住 qián，其餘才轉「干」）
    # 只做逐字轉換會把這兩類都轉錯，所以兩張表都要帶。
    phrases: dict[str, str] = {}
    for name in ("TWVariantsRevPhrases.txt", "TSPhrases.txt"):
        table = dictionary / name
        if not table.is_file():
            continue
        for line in table.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("\t")
            key, value = key.strip(), value.split()
            if key and value:
                phrases[key] = value[0]
    phrase_path = out.with_name("t2s_phrases.json")
    phrase_path.write_text(json.dumps(phrases, ensure_ascii=False, indent=0, sort_keys=True),
                           encoding="utf-8")
    print(f"詞組規則 {len(phrases):,} 條 → {phrase_path.name}")

    # 轉換的目的就是消掉位元組碎片，所以把「哪些字仍然是碎片」也帶出來，
    # 執行端才能在轉完之後回報還有哪幾個字沒救到（多半是台文專用字：佮、拚、囝）。
    frag_path = out.with_name("t2s_fragments.json")
    still = sorted(c for c in chars if fragments(c))
    frag_path.write_text(json.dumps(still, ensure_ascii=False), encoding="utf-8")
    print(f"碎片字 {len(still):,} 個 → {frag_path.name}")

    print(f"掃描 {len(chars):,} 個漢字")
    print(f"  可安全轉換   {len(safe):,}")
    print(f"  沒有碎片     {len(skipped['沒有碎片']):,}")
    print(f"  簡體仍碎片   {len(skipped['簡體仍碎片']):,}")
    
    print(f"\n寫出 {out}")
    print("  樣本: " + " ".join(f"{k}→{v}" for k, v in list(safe.items())[:12]))

    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", encoding="utf-8") as sink:
            for reason, items in skipped.items():
                if reason == "沒有碎片":
                    sink.write(f"# {reason}: {len(items):,} 字（略）\n\n")
                    continue
                sink.write(f"# {reason}: {len(items):,} 字\n")
                sink.write("".join(f"  {x}\n" for x in items))
                sink.write("\n")
        print(f"排除清單 {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
