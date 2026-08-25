#!/usr/bin/env python3
"""Gemini 當消歧層——只准它挑編號，不准它寫讀音。

語料學出的預設（taigi_annotate.by_corpus）已經有 85.6%，不看句子的天花板是
94.2%。Gemini 要做的是剩下那些**必須看句子**的：

    我     gua2 511 次 / guan2 52 次     單複數
    媽媽   a1-bo2 60 / ma1-ma1 18        語域
    在     ti7 48 / tsai7 21             語法位置

輸出一律驗證：不是合法編號就退回語料預設。壞掉的方向是安全的。

    .venv-gemini/bin/python tools/taigi_gemini_backend.py --quiz quiz.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

PROMPT = """你是台語讀音選擇器。下面每一題給一句華語，和其中一個詞的台語讀音候選。
依照句子的意思，判斷這個詞在這一句該用哪個讀音。

規則：
- 只回編號，不要自己寫讀音。
- 注意單複數（我 gua2／阮 guan2）、語域（阿母／媽媽）、語法位置。
- 不確定就填 null。填 null 會退回語料統計出來的預設，那是安全的。

{items}

只輸出 JSON 陣列，第 i 個元素是第 i 題的編號或 null。不要解釋。"""


def load_env() -> dict[str, str]:
    env_file = ROOT / "docker/testbench/.env"
    env = dict(re.findall(r"^(\w+)=(.*)$", env_file.read_text(encoding="utf-8"), re.M))
    return {k: v.strip() for k, v in env.items()}


def ask(questions: list[tuple[str, str, list[str]]], batch: int = 25) -> list[int | None]:
    """questions = [(句子, 詞, 候選), ...] → 每題一個編號或 None。"""
    from google import genai
    from google.genai import types

    env = load_env()
    client = genai.Client(api_key=env["GEMINI_API_KEY"])
    model = env.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
    level = env.get("GEMINI_THINKING_LEVEL", "medium")

    out: list[int | None] = []
    for start in range(0, len(questions), batch):
        chunk = questions[start:start + batch]
        listing = "\n".join(
            f"{i}. 句子：{s}\n   詞：{w}\n   候選：" +
            "  ".join(f"[{j}]{k}" for j, k in enumerate(cs))
            for i, (s, w, cs) in enumerate(chunk))
        resp = client.models.generate_content(
            model=model,
            contents=PROMPT.format(items=listing),
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=level),
                response_mime_type="application/json",
            ),
        )
        try:
            picked = json.loads(resp.text)
        except Exception:
            m = re.search(r"\[.*\]", resp.text or "", re.S)
            picked = json.loads(m.group()) if m else []
        picked = list(picked) + [None] * (len(chunk) - len(picked))
        out.extend(picked[:len(chunk)])
        print(f"    …{min(start + batch, len(questions))}/{len(questions)}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiz", required=True, help="TSV：編號 句子 詞=X 候選")
    ap.add_argument("--gold", help="答案 JSON，給了就評分")
    a = ap.parse_args()

    questions, cands_all = [], []
    for line in Path(a.quiz).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _, sent, word, opts = line.split("\t", 3)
        cs = re.findall(r"\[\d+\]([A-Za-z0-9\-]+)", opts)
        questions.append((sent, word.removeprefix("詞="), cs))
        cands_all.append(cs)

    picks = ask(questions)
    if not a.gold:
        print(json.dumps(picks))
        return 0

    gold = json.load(open(a.gold, encoding="utf-8"))
    gold = [g["gold"] if isinstance(g, dict) else g for g in gold]
    right = invalid = 0
    wrong = []
    for (sent, word, cs), p, g in zip(questions, picks, gold):
        if not isinstance(p, int) or isinstance(p, bool) or not 0 <= p < len(cs):
            invalid += 1
            continue
        if cs[p] == g:
            right += 1
        else:
            wrong.append((word, cs[p], g, sent[:26]))
    n = len(gold)
    print(f"  Gemini  {right}/{n} = {right/n*100:.0f}%   （無效輸出 {invalid} 題，會退回預設）")
    for w, got, want, s in wrong[:12]:
        print(f"    {w:<4} 挑 {got:<16} 答案 {want:<16} 「{s}」")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
