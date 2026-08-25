"""台文對照表的分層儲存與回饋紀錄。

表分三層，查詢由上往下，**下層重建不會動到上層**：

    hanji_overrides.json   人工／回饋修正    最優先，只有人能改
    hanji_llm.json         LLM 補的空缺      標了來源與時間，可整層撤掉
    hanji_table.json       語料＋辭典生成    可隨時重建

分層是因為 LLM 會錯。實測有一句它把查表已經答對的「舊年」改回華語的「去年」。
讓它直接覆蓋底表的話，錯了就找不回來，也分不出哪些是它改的。

資料放在可寫的目錄（容器裡 repo 是唯讀掛載），路徑用 TAIGI_DATA 指定。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

DATA = Path(os.environ.get("TAIGI_DATA", "/data/taigi"))
REPO = Path(__file__).resolve().parent.parent / "workspace/chhoetaigi"

BASE = REPO / "hanji_table.json"            # 生成的，跟著 repo 走
# 底表有幾個版本可以切換，方便聽測比較。上面兩層（人工修正、LLM 補的）不受影響。
# 只留兩個能拿來比的版本。中間那幾版（v2/v3/v4）都有已知的錯，移到
# archive/，留著當「為什麼現在這樣做」的證據，不列進選單。
BASES = {
    "v5": REPO / "hanji_table.v5-multiword-only.json",       # 對齊法，精確 64%／召回 23%
    "v1": REPO / "hanji_table.v1-chhoetaigi-naer.json",      # 辭典法，精確 42%／召回 25%
}
LLM = DATA / "hanji_llm.json"               # LLM 補的
OVERRIDES = DATA / "hanji_overrides.json"   # 人工修正
UNKNOWN = DATA / "hanji_unknown.jsonl"      # 查不到的詞，等著離線補
FEEDBACK = DATA / "feedback.jsonl"          # 點讚點爛
DISMISSED = DATA / "dismissed.json"         # 看過但不採用的建議


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _append(path: Path, row: dict) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except OSError:
        return False


def layers(version: str | None = None) -> tuple[dict, dict, dict]:
    """version 指定底表版本；None 用目前生效的 hanji_table.json。"""
    path = BASES.get(version, BASE) if version else BASE
    return _load(path), _load(LLM), _load(OVERRIDES)


def merged(version: str | None = None) -> dict[str, str]:
    """三層合起來的查詢表。上層蓋下層。"""
    base, llm, over = layers(version)
    out = dict(base)
    out.update({w: v["hanji"] if isinstance(v, dict) else v for w, v in llm.items()})
    out.update({w: v["hanji"] if isinstance(v, dict) else v for w, v in over.items()})
    return out


def source_of(word: str) -> str:
    """這個詞的譯法是哪一層給的。除錯與審查用。"""
    base, llm, over = layers()
    if word in over:
        return "override"
    if word in llm:
        return "llm"
    return "table" if word in base else "none"


def log_unknown(text: str, unknown: list[str]) -> bool:
    return _append(UNKNOWN, {"t": int(time.time()), "text": text, "unknown": unknown})


def log_feedback(*, verdict: str, text: str, sent: str, mode: str,
                 changed: list, blamed: list | None = None,
                 suggested: dict | None = None) -> bool:
    """verdict 是 up 或 down。

    blamed 是使用者指出哪幾個詞不對；沒指出就是整句未歸因。未歸因也有用——
    同一個詞出現在很多爛句子、很少好句子，統計上就浮得出來。

    suggested 是 {華語詞: 正確的台語寫法}，**選填**。給了就直接知道答案，
    不必再問 LLM 或猜；沒給也照樣記錄，那個詞會排進待處理清單。
    """
    return _append(FEEDBACK, {
        "t": int(time.time()), "verdict": verdict, "mode": mode,
        "text": text, "sent": sent,
        "changed": [list(c) for c in changed], "blamed": blamed or [],
        "suggested": suggested or {},
    })


def dismissed() -> dict:
    """看過但不採用的建議。key 是「華語詞→建議」，值是排除的原因與時間。

    只是隱藏，原始回饋還在 feedback.jsonl——那是紀錄，不該因為看過就消失。
    排除是可以撤銷的。
    """
    return _load(DISMISSED)


def dismiss(word: str, suggest: str, *, why: str = "", undo: bool = False) -> None:
    d = _load(DISMISSED)
    key = f"{word}→{suggest}"
    if undo:
        d.pop(key, None)
    else:
        d[key] = {"at": int(time.time()), "why": why}
    DISMISSED.parent.mkdir(parents=True, exist_ok=True)
    DISMISSED.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def put_llm(word: str, hanji: str, *, note: str = "") -> None:
    """LLM 補一個詞。記來源與時間，之後審查或整批撤掉都找得到。"""
    d = _load(LLM)
    d[word] = {"hanji": hanji, "by": "llm", "at": int(time.time()), "note": note}
    LLM.parent.mkdir(parents=True, exist_ok=True)
    LLM.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def put_override(word: str, hanji: str, *, why: str = "") -> None:
    d = _load(OVERRIDES)
    d[word] = {"hanji": hanji, "by": "human", "at": int(time.time()), "why": why}
    OVERRIDES.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
