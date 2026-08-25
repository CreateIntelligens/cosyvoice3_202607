"""/admin —— 維護台文對照表。跟試聽台和 /v1 完全分開。

為什麼要分開：

    /api/*    試聽台自己的頁面在用，沒有權杖
    /v1/*     給 agent 呼叫，權杖是 TTS_API_KEY，只能合成
    /admin/*  改的是所有人共用的對照表，權杖是 ADMIN_API_KEY

拿 /v1 權杖打 /admin 會被擋。合成的權限跟「改變全站發音」的權限不該是同一把
鑰匙——agent 拿得到前者，不該順便拿到後者。

沒設 ADMIN_API_KEY 的話整組端點直接不掛載，不是回 401 而是不存在。

流程跟 tools/taigi_fill_table.py 一樣，刻意分成三步：

    GET  /admin/table/pending    看累積了哪些查不到的詞
    POST /admin/table/fill       問 LLM。預設 dry=true 只回建議，不寫入
    POST /admin/table/fill       dry=false 才真的寫進 hanji_llm.json
    DELETE /admin/table/llm      整層撤掉

不自動補是因為 LLM 會錯得很像樣——實測它把「鄰近」補成「體近」、「氣象署」
補成「天氣署」，兩個都不是詞。
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
import urllib.request

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

sys.path.insert(0, "/opt/CosyVoice/tools")

ADMIN_KEY = os.environ.get("ADMIN_API_KEY", "")
bearer = HTTPBearer(auto_error=False)          # auto_error=False 才有 Swagger 的 Authorize 鈕

router = APIRouter(prefix="/admin", tags=["admin：對照表維護"])


def require_admin(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if not ADMIN_KEY:
        raise HTTPException(503, "ADMIN_API_KEY 未設定，維護端點停用")
    if cred is None or cred.credentials != ADMIN_KEY:
        raise HTTPException(401, "Authorization 缺少或不正確（這裡要的是 ADMIN_API_KEY，"
                                 "不是 /v1 的 TTS_API_KEY）")


class Pending(BaseModel):
    blamed: list[dict] = Field(..., description="點爛時被指名的詞，優先處理")
    unknown: list[dict] = Field(..., description="查表查不到、原字送出去的詞")
    counts: dict = Field(..., description="各層的詞數")


class FillRequest(BaseModel):
    dry: bool = Field(True, description="true 只回建議不寫入。**預設 true**")
    top: int = Field(50, ge=1, le=200, description="處理出現次數最多的前幾個")
    words: list[str] | None = Field(None, description="指定要補哪些詞；不給就用累積的未知詞")


class Proposal(BaseModel):
    word: str
    hanji: str | None
    accepted: bool
    reason: str


class FillResult(BaseModel):
    dry: bool
    asked: int
    accepted: int
    rejected: int
    proposals: list[Proposal]


class OverrideRequest(BaseModel):
    word: str = Field(..., description="華語詞")
    hanji: str = Field(..., description="台語漢字寫法")
    why: str = Field("", description="為什麼這樣改，之後審查用")


def _store():
    import taigi_store
    return taigi_store


@router.get("/table/pending", response_model=Pending,
            summary="看累積了哪些查不到的詞")
def pending(_: None = Depends(require_admin)):
    import taigi_fill_table as F
    st = _store()
    base, llm, over = st.layers()
    have = st.merged()
    return Pending(
        blamed=[{"word": w, "times": n, "current": have.get(w)}
                for w, n in F.blamed().most_common(50)],
        unknown=[{"word": w, "times": n} for w, n in F.pending().most_common(100)],
        counts={"table": len(base), "llm": len(llm), "override": len(over),
                "merged": len(have)},
    )


@router.post("/table/fill", response_model=FillResult,
             summary="問 LLM 補未知詞（預設只回建議，不寫入）")
def fill(req: FillRequest, _: None = Depends(require_admin)):
    import taigi_fill_table as F
    words = req.words or [w for w, _ in F.pending().most_common(req.top)]
    if not words:
        return FillResult(dry=req.dry, asked=0, accepted=0, rejected=0, proposals=[])
    try:
        got = F.ask(words)
    except Exception as e:
        raise HTTPException(502, f"LLM 呼叫失敗：{type(e).__name__}")

    props, ok = [], {}
    for w in words:
        raw = got.get(w)
        good = F.verify(w, raw)
        if good:
            ok[w] = good
            props.append(Proposal(word=w, hanji=good, accepted=True, reason="辭典查得到"))
        else:
            props.append(Proposal(word=w, hanji=raw if isinstance(raw, str) else None,
                                  accepted=False,
                                  reason="LLM 說不用換" if not raw else "查無此詞，不收"))
    if not req.dry:
        st = _store()
        for w, h in ok.items():
            st.put_llm(w, h, note="admin-api")
    return FillResult(dry=req.dry, asked=len(words), accepted=len(ok),
                      rejected=len(words) - len(ok), proposals=props)


@router.delete("/table/llm", summary="撤掉 LLM 補的那一層（底表與人工修正不受影響）")
def revert(_: None = Depends(require_admin)):
    st = _store()
    n = len(st._load(st.LLM))
    st.LLM.unlink(missing_ok=True)
    return {"removed": n}


@router.put("/table/override", summary="人工修正一個詞（最優先，不會被自動流程覆蓋）")
def override(req: OverrideRequest, _: None = Depends(require_admin)):
    st = _store()
    st.put_override(req.word, req.hanji, why=req.why)
    return {"ok": True, "word": req.word, "hanji": req.hanji, "layer": "override"}


class DismissRequest(BaseModel):
    word: str
    suggest: str
    why: str = Field("", description="為什麼不採用")
    undo: bool = Field(False, description="true 表示撤銷排除")


@router.post("/suggestions/dismiss", summary="排除一則建議（不採用，但紀錄留著）")
def dismiss(req: DismissRequest, _: None = Depends(require_admin)):
    """排除只是隱藏，原始回饋還在 feedback.jsonl——那是紀錄，不該因為看過就消失。

    所以可以撤銷（undo=true），也不會影響統計。
    """
    st = _store()
    st.dismiss(req.word, req.suggest, why=req.why, undo=req.undo)
    return {"ok": True, "word": req.word, "suggest": req.suggest,
            "state": "撤銷排除" if req.undo else "已排除"}


@router.get("/table/lookup/{word}", summary="查一個詞目前的譯法來自哪一層")
def lookup(word: str, _: None = Depends(require_admin)):
    st = _store()
    base, llm, over = st.layers()

    def unwrap(v):
        return v["hanji"] if isinstance(v, dict) else v

    return {
        "word": word,
        "effective": st.merged().get(word),
        "source": st.source_of(word),
        "layers": {"override": unwrap(over.get(word)), "llm": unwrap(llm.get(word)),
                   "table": base.get(word)},
    }


@router.get("/feedback/compare",
            summary="同一句在不同模式下的評價比較——回答「轉換有沒有比較好」")
def compare(_: None = Depends(require_admin)):
    """只有成對的評價答得出「轉換值不值得」。

    單看轉換版的評價，只知道它好不好；跟 none 擺在一起才知道它比不轉換好還是壞。
    **轉換讓事情變糟的句子最有價值**——那直接指出表裡有壞條目，而且嫌疑範圍
    就是那句換過的幾個詞，不必人工歸因。
    """
    st = _store()
    by_text: dict[str, dict] = {}
    try:
        for line in st.FEEDBACK.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            kind = "none" if row.get("mode", "none") == "none" else \
                   ("hanji" if row["mode"].startswith("hanji") else "tailo")
            slot = by_text.setdefault(row["text"], {})
            # 同一句同一種模式評很多次時，以最後一次為準
            slot[kind] = {"verdict": row["verdict"], "mode": row["mode"],
                          "sent": row["sent"], "changed": row.get("changed", [])}
    except OSError:
        pass

    better, worse, unpaired = [], [], []
    suspect: collections.Counter = collections.Counter()
    for text, slots in by_text.items():
        base = slots.get("none")
        for kind in ("hanji", "tailo"):
            conv = slots.get(kind)
            if not conv:
                continue
            if not base:
                unpaired.append({"text": text, "mode": conv["mode"],
                                 "verdict": conv["verdict"]})
                continue
            row = {"text": text, "mode": conv["mode"], "sent": conv["sent"],
                   "none": base["verdict"], "converted": conv["verdict"],
                   "changed": conv["changed"]}
            if base["verdict"] == "down" and conv["verdict"] == "up":
                better.append(row)
            elif base["verdict"] == "up" and conv["verdict"] == "down":
                worse.append(row)
                # 轉換把好的弄壞了，嫌疑就在這句換過的詞裡
                suspect.update(p[0] for p in conv["changed"] if p)
    return {
        "summary": {"轉換變好": len(better), "轉換變壞": len(worse),
                    "只評了轉換版": len(unpaired), "評過的句子": len(by_text)},
        "轉換變壞": worse[:20],
        "轉換變好": better[:20],
        "嫌疑詞": [{"word": w, "times": n} for w, n in suspect.most_common(20)],
        "只評了一邊": unpaired[:20],
    }


@router.get("/feedback", summary="點讚點爛的統計")
def feedback(limit: int = 30, _: None = Depends(require_admin)):
    st = _store()
    up = down = 0
    per_word = collections.defaultdict(lambda: {"up": 0, "down": 0, "blamed": 0})
    try:
        for line in st.FEEDBACK.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            v = row.get("verdict")
            up, down = up + (v == "up"), down + (v == "down")
            for pair in row.get("changed", []):
                if pair:
                    per_word[pair[0]][v] += 1
            for w in row.get("blamed", []):
                per_word[w]["blamed"] += 1
    except OSError:
        pass
    # 被指名的排前面，其次是負評多的——那些是最該看的
    ranked = sorted(per_word.items(),
                    key=lambda kv: (-kv[1]["blamed"], -kv[1]["down"], kv[1]["up"]))
    return {"total": {"up": up, "down": down},
            "words": [{"word": w, **s} for w, s in ranked[:limit]]}
