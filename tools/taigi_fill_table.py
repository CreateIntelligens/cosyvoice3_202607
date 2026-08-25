#!/usr/bin/env python3
"""把累積的未知詞拿去問 LLM，補進表裡。**手動跑，不是自動的。**

線上永遠不等 LLM：查表查不到就原字送出（等於現在不轉換的行為），同時把詞記
下來。累積一批之後跑這支，補進 hanji_llm.json。下次就查得到，零延遲。

自動補會讓 LLM 悄悄改變系統行為，而它會錯——實測它把「鄰近」補成「體近」、
「氣象署」補成「天氣署」，兩個都不是詞。所以：

    1. --list    先看累積了什麼
    2. --dry     問 LLM 但不寫入，看它給什麼
    3. --apply   確認之後才寫進 hanji_llm.json

寫進去的是**獨立一層**，不會覆蓋語料生成的底表，也不會覆蓋人工修正。
補錯了可以 --revert 整層撤掉。

點爛的紀錄也在這裡用：被指名的詞會列在 --list 最上面，那些是已知有問題的。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taigi_store as store  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPT = """下面每個華語詞要換成臺灣台語的漢字寫法，交給台語 TTS 朗讀。

規則：
- 用教育部推薦用字。
- 只給台語真的這樣講的詞。**沒把握就填 null**，填 null 會保留華語原字，那是安全的；
  編一個不存在的詞比不換更糟。
- 台語跟華語同字的（例如「氣象」「紀錄」）也填 null，不用換。

詞：
{words}

只輸出 JSON：{{"華語詞": "台語漢字" 或 null, ...}}"""


def pending() -> collections.Counter:
    """還沒補進表的未知詞，依出現次數排序。"""
    have = store.merged()
    c = collections.Counter()
    try:
        for line in store.UNKNOWN.read_text(encoding="utf-8").splitlines():
            for w in json.loads(line).get("unknown", []):
                if w not in have:
                    c[w] += 1
    except OSError:
        pass
    return c


def blamed() -> collections.Counter:
    """被點爛時指名的詞。這些是已知有問題的，優先處理。"""
    c = collections.Counter()
    try:
        for line in store.FEEDBACK.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("verdict") == "down":
                c.update(row.get("blamed", []))
    except OSError:
        pass
    return c


def ask(words: list[str]) -> dict:
    env = dict(re.findall(r"^(\w+)=(.*)$",
                          (ROOT / "docker/testbench/.env").read_text(encoding="utf-8"), re.M))
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{env['GEMINI_MODEL'].strip()}:generateContent?key={env['GEMINI_API_KEY'].strip()}")
    out = {}
    for i in range(0, len(words), 40):
        chunk = words[i:i + 40]
        body = json.dumps({
            "contents": [{"parts": [{"text": PROMPT.format(words="\n".join(chunk))}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }).encode()
        with urllib.request.urlopen(
                urllib.request.Request(url, body, {"Content-Type": "application/json"}),
                timeout=90) as r:
            txt = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            out.update(json.loads(m.group()))
        print(f"  …{min(i + 40, len(words))}/{len(words)}", file=sys.stderr)
    return out


def verify(word: str, hanji) -> str | None:
    """LLM 給的寫法要在辭典裡查得到，否則不收。"""
    if not isinstance(hanji, str) or not hanji.strip():
        return None
    hanji = hanji.strip()
    if hanji == word or not re.fullmatch(r"[一-鿿]+", hanji):
        return None
    from taigi_annotate import taigi_words
    return hanji if hanji in taigi_words() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="只列出累積了哪些未知詞")
    ap.add_argument("--dry", action="store_true", help="問 LLM 但不寫入")
    ap.add_argument("--apply", action="store_true", help="確認後寫進 hanji_llm.json")
    ap.add_argument("--revert", action="store_true", help="撤掉 LLM 補的那一層")
    ap.add_argument("--top", type=int, default=100, help="一次處理幾個（依出現次數）")
    a = ap.parse_args()

    if a.revert:
        store.LLM.unlink(missing_ok=True)
        print("  已撤掉 hanji_llm.json，底表與人工修正不受影響")
        return 0

    want, bad = pending(), blamed()
    if a.list or not (a.dry or a.apply):
        if bad:
            print(f"  點爛時被指名的詞（{len(bad)} 個，優先處理）：")
            for w, n in bad.most_common(20):
                print(f"    {w}  被指 {n} 次   目前譯法 {store.merged().get(w, '（無）')}")
        print(f"\n  未知詞 {len(want)} 個，出現最多的：")
        for w, n in want.most_common(30):
            print(f"    {w}  {n} 次")
        return 0

    words = [w for w, _ in want.most_common(a.top)]
    if not words:
        print("  沒有累積的未知詞")
        return 0
    got = ask(words)
    ok, rejected = {}, []
    for w in words:
        v = verify(w, got.get(w))
        (ok.__setitem__(w, v) if v else rejected.append((w, got.get(w))))
    print(f"\n  收 {len(ok)} 個，退 {len(rejected)} 個")
    for w, h in list(ok.items())[:30]:
        print(f"    ✅ {w} → {h}")
    for w, h in rejected[:15]:
        why = "查無此詞" if h else "LLM 說不用換"
        print(f"    ✗  {w} → {h}  （{why}）")
    if a.apply:
        for w, h in ok.items():
            store.put_llm(w, h, note="batch")
        print(f"\n  已寫進 {store.LLM}（獨立一層，--revert 可整批撤掉）")
    else:
        print("\n  這是 --dry，沒有寫入。確認之後加 --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
