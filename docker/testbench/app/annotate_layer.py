"""試聽台的標音層：華語進來，整句換成台羅再送給 v7。

三種模式讓耳朵直接比：

    none     原字送進去（現在的行為）
    corpus   語料學出的讀音。naer 的 107,292 句雙寫法統計出來的，85.6%
    gemini   多義詞交給 Gemini 挑。**評測上比 corpus 差很多**（隨機 60 題
             95% → 60%），錯法是台語詞彙知識而不是理解句子——「一」的
             it4/it8 它連錯四題。放這裡是因為評測量的是「跟 naer 的寫法
             一不一致」，不是自然度，而那要用聽的。

整句換或整句不換，不做混排：量過 10 個 seed，漢字夾在台羅中間 1/10 會幻覺
（3.5 秒的句子生出 21.6 秒），純漢字與全台羅都是 0/10。
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, "/opt/CosyVoice/tools")

MODES = ("none", "corpus", "gemini", "hanji", "hanji_llm")
_gemini_cache: dict[str, dict] = {}
_GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:generateContent?key={key}")
_PROMPT = """這句華語要唸成台語。用哪些台語詞已經決定好了，你只要判斷
**同一個詞的多種讀音**在這一句該用哪一個。

台語很多字有文讀與白話兩種唸法，用法不同：

    數字唸號碼、年份、序數 → 文讀     一九四四 it4、第一 te7-it4
    數字用來計量、指東西   → 白話     一個 tsit8、六隻 lak8
    書面語、成語、專有名詞 → 文讀     人口 jin5-khau2
    日常口語               → 白話     彼个人 lang5

句子：{sentence}

{items}

規則：
- 只能從候選裡選編號，不可以自己寫讀音。
- 這些候選是同一個詞的不同唸法，不是不同的詞——選錯不會改變意思，只會唸錯。
- 判斷不出來就填 null，會退回預設，那是安全的。

只輸出 JSON：{{"詞": 編號或 null, ...}}"""


_gloss_cache: dict[str, str] | None = None


def _kip_gloss() -> dict[str, str]:
    """台羅 → 台語漢字。沒有這個，Gemini 只看得到羅馬字，等於盲猜。

    從辭典的快取建，取最短的寫法（最常見的本字，不是異體）。
    """
    global _gloss_cache
    if _gloss_cache is not None:
        return _gloss_cache
    _gloss_cache = {}
    try:
        import csv
        path = "/opt/CosyVoice/workspace/chhoetaigi/ChhoeTaigi_TaihoaSoanntengTuichiautian.csv"
        from taigi_annotate import add_tone_digits
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                kip = add_tone_digits((row.get("KipInput") or "").strip())
                han = (row.get("HanLoTaibunKip") or "").strip()
                if not kip or not han:
                    continue
                old = _gloss_cache.get(kip)
                if old is None or len(han) < len(old):
                    _gloss_cache[kip] = han
    except Exception:
        pass                                     # 沒有對照表就退回只給羅馬字
    return _gloss_cache


def _multi_reading_choices(text: str, lex) -> list[tuple[str, list[str]]]:
    """挑出句子裡「台語詞形已定、但那個詞有多種唸法」的詞。"""
    from taigi_annotate import by_corpus, segment
    multi = _multi_readings()
    out = []
    for word, readings in segment(text, lex):
        if not readings:
            continue
        default = by_corpus(word, readings)
        alts = multi.get(word) or []
        cands = [r for r, _ in alts if not r[:1].isupper()] if alts else []
        if default not in cands:
            cands = [default] + cands
        if len(cands) > 1:
            out.append((word, cands[:5]))
    return out


_multi_cache: dict | None = None


def _multi_readings() -> dict:
    """台語詞 → [(讀音, 文讀/白話), ...]。教育部辭典裡收了多種唸法的那些。"""
    global _multi_cache
    if _multi_cache is None:
        try:
            with open("/opt/CosyVoice/workspace/moe_sutian/multi_readings.json",
                      encoding="utf-8") as f:
                _multi_cache = json.load(f)
        except Exception:
            _multi_cache = {}
    return _multi_cache


def _gemini_pick_all(sentence: str, choices: list[tuple[str, list[str]]],
                     gloss=None) -> dict[str, int]:
    """問 Gemini，回傳 詞→編號。任何失敗都回空的，呼叫端自動退回 corpus。"""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key or not choices:
        return {}
    gloss = gloss or {}
    tags = {r: t for alts in _multi_readings().values() for r, t in alts if t}
    listing = "\n".join(
        f"{w}：" + "  ".join(
            f"[{i}]{k}（{tags[k]}）" if k in tags else f"[{i}]{k}"
            for i, k in enumerate(ks))
        for w, ks in choices)
    body = json.dumps({
        "contents": [{"parts": [{"text": _PROMPT.format(sentence=sentence, items=listing)}]}],
        # 不設的話同一句問兩次會拿到不同答案，兩個 seed 的文字就不一樣，A/B 白做。
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }).encode()
    url = _GEMINI_URL.format(model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                             key=key)
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = json.load(r)
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group()) if m else {}


_HANJI_PROMPT = """辭典已經把這句華語的大部分詞換成台語漢字了，但有幾個詞查不到。
你只要補那幾個詞，其他一個字都不要動。

已經換好的（**不要改動這些**）：
{gloss}

查不到、要你補的詞：{missing}

目前的句子：{s}

規則：
- 只把「要你補的詞」換成台語說法，其餘原樣保留。
- 用教育部推薦用字。虛詞：把→共、被→予、給→予、他→伊、了→矣、什麼→啥物。
- 不改句子結構、不增減資訊、不解釋。標點保留。

只輸出整句結果，不要任何其他文字。"""


def _hanji_llm(converted: str, changed: list, missing: list[str]) -> str | None:
    """只補查表補不到的詞。

    先前是整句丟給它、查表結果只當提示——結果它會**覆蓋查表已經答對的詞**
    （實測有一句把對的「舊年」改回華語的「去年」）。改成只補洞之後，查表答對
    的部分它碰不到，而且整句都查得到時完全不用呼叫。
    """
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key or not missing:
        return None
    gloss = "\n".join(f"  {w} → {h}" for w, h in dict(changed).items()) or "  （無）"
    body = json.dumps({
        "contents": [{"parts": [{"text": _HANJI_PROMPT.format(
            gloss=gloss, missing="、".join(missing), s=converted)}]}],
        "generationConfig": {"temperature": 0},
    }).encode()
    url = _GEMINI_URL.format(model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                             key=key)
    with urllib.request.urlopen(
            urllib.request.Request(url, body, {"Content-Type": "application/json"}),
            timeout=30) as r:
        out = json.load(r)["candidates"][0]["content"]["parts"][0]["text"].strip()
    # LLM 有機會亂加東西。長度差太多就是改寫過頭，不採用。
    if not out or not (0.5 <= len(out) / max(len(converted), 1) <= 1.8):
        return None
    bad = _unverifiable(converted, out)
    if bad:
        # 它會編出不存在的詞（實測「鄰近→體近」「氣象署→天氣署」）。
        # 編造的台文比留著華語原字更糟——原字至少模型唸得出意思。
        print(f"  台文 LLM 產生查無此詞的寫法，整句退回：{bad}", file=sys.stderr)
        return None
    return out


def _known_words() -> frozenset[str]:
    """教育部收錄的台語詞目＋台華線頂的台語寫法。用來擋掉 LLM 編的詞。"""
    global _known_cache
    if _known_cache is None:
        from taigi_annotate import taigi_words
        words = set(taigi_words())
        words |= {h for h in _kip_gloss().values() if h}
        _known_cache = frozenset(words)
    return _known_cache


_known_cache: frozenset[str] | None = None


def _unverifiable(before: str, after: str) -> list[str]:
    """LLM 換出來、但辭典查不到的詞。長度 2 以上的漢字串才檢查。"""
    import re as _re
    known = _known_words()
    old = set(_re.findall(r"[一-鿿]{2,}", before))
    out = []
    for chunk in _re.findall(r"[一-鿿]{2,}", after):
        if chunk in old or chunk in known:
            continue
        # 整串查不到時，看它是不是由查得到的詞接起來的（斷詞邊界不一定對齊）
        if not any(chunk[i:j] in known for i in range(len(chunk))
                   for j in range(i + 2, len(chunk) + 1)):
            out.append(chunk)
    return out


def apply(text: str, mode: str, table_version: str | None = None,
          extra: dict[str, str] | None = None) -> tuple[str, str, list]:
    """回傳（送進模型的文字, 給人看的說明）。任何失敗都退回原字。"""
    if mode == "none":
        return text, "none", []

    # 台文路線：輸出仍是漢字，不轉台羅。v6 用 naer 290 小時訓練，而 naer 的原文
    # 就是台語漢字——所以這是模型的訓練分布本身，語調跟餵華語一樣自然。
    if mode in ("hanji", "hanji_llm"):
        try:
            from taigi_hanji import convert
        except Exception as e:
            return text, f"unavailable:{type(e).__name__}", []
        # extra 是這一次請求專用的覆寫，不寫進任何一層——用來試聽「改了會怎樣」，
        # 聽完滿意再真的寫進人工修正層。
        out, changed, unknown = convert(text, log_unknown=not extra,
                                        version=table_version, extra=extra)
        if mode == "hanji" or not unknown:
            # 整句都查得到就不必呼叫 LLM——零延遲、零成本，也沒有被改壞的風險
            tail = "" if not unknown else f"，{len(unknown)}詞查無"
            ver = f"[{table_version}]" if table_version else ""
            return out, f"hanji{ver}:換了{len(changed)}詞{tail}", changed
        if out in _gemini_cache:
            got = _gemini_cache[out]
        else:
            try:
                got = _hanji_llm(out, changed, unknown)
            except Exception:
                got = None
            _gemini_cache[out] = got
        if got:
            return got, f"hanji_llm:查表{len(changed)}詞＋LLM補{len(unknown)}詞", changed
        return out, f"hanji_llm-failed→hanji:換了{len(changed)}詞", changed
    try:
        from taigi_annotate import annotate_clause_wise, by_corpus, lexicon, segment
    except Exception as e:                       # 辭典沒掛進來就別擋住合成
        return text, f"unavailable:{type(e).__name__}", []

    picked: dict[str, str] = {}
    if mode == "gemini":
        lex = lexicon()
        # 只問「同一個詞的多種唸法」，不問「台語用哪個詞」。
        # 後者是知識，語料與教育部辭典裡有、模型腦袋裡沒有——先前拿那個問它，
        # 隨機 60 題 95% 掉到 60%，「一」的 it4/it8 連錯四題。
        choices = _multi_reading_choices(text, lex)
        if text in _gemini_cache:
            picked = _gemini_cache[text]
        else:
            try:
                idxs = _gemini_pick_all(text, choices, gloss=_kip_gloss())
            except Exception as e:
                return text, f"gemini-failed:{type(e).__name__}", []
            # 編號是對 choices 清單的，趁還看得到那份清單就換成讀音本身
            picked = {}
            for word, cands in choices:
                i = idxs.get(word)
                if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(cands):
                    picked[word] = cands[i]
            _gemini_cache[text] = picked

    def pick(word: str, readings: list[str]) -> str:
        # Gemini 只在文白異讀上有發言權；它挑的讀音不必在台華線頂的候選裡，
        # 因為那份清單對「讀音」不權威（影響只收 ing2-hiang2，教育部是 hiong2）。
        got = picked.get(word)
        return got if got else by_corpus(word, readings)

    out, changed = annotate_clause_wise(text, pick=pick)
    if out.strip() == text.strip():
        return text, f"{mode}:整句都查不到讀音，原字送出", []
    return out, f"{mode}:換了{len(changed)}詞", [(w, k) for w, k, _ in changed]
