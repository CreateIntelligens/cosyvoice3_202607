"""試聽台的標音層：華語進來，換成台語漢字或台羅之後再送給 v7。

七種模式，選單上只露出四種：

    none          原字送進去
    hanji         華語詞換成台語漢字，查對照表。**選單上的預設路線**
    hanji_direct  同上，但整句交給 LLM，完全不查表
    corpus        換成台羅。讀音由 naer 的 107,292 句雙寫法統計選出，85.6%
    tailo_direct  同上，但整句交給 LLM

    hanji_llm     查表之後，只把查不到的詞交給 LLM 補。**選單上已移除**
    gemini        台羅的多義詞交給 LLM 挑文白。**選單上已移除**——評測比
                  corpus 差很多（隨機 60 題 95% → 60%），錯法是台語詞彙
                  知識而不是理解句子。程式碼保留，API 仍可指定。

台文那條是主力：v6/v7 用 naer 290 小時訓練，而 naer 的原文就是台語漢字，
所以那是模型的訓練分布本身。台羅只有 4.62 小時，唸起來偏硬。

**整句換或整句不換，不做混排**：量過 10 個 seed，漢字夾在台羅中間 1/10 會
幻覺（3.5 秒的句子生出 21.6 秒），純漢字與全台羅都是 0/10。這也是為什麼
_direct 那兩條驗證不過時是整句退回，而不是只換掉壞掉的那個詞——LLM 的輸出
跟原文沒有對齊資訊，換一半比不換更糟。

_direct 兩條的退回是分層的：LLM → 查表 → 原字。退到查表而不是原字，是因為
查表只要 7 毫秒而且安全，它的弱點是涵蓋率不是正確性。
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import unicodedata
import urllib.request
from pathlib import Path

# 容器把 repo 掛在 /opt/CosyVoice，但本機直接跑時不是。用環境變數指，
# 兩種情境都能用。
REPO = Path(os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice"))
sys.path.insert(0, str(REPO / "tools"))

MODES = ("none", "corpus", "gemini", "hanji", "hanji_llm",
         "hanji_direct", "tailo_direct")
# ── 直接問 LLM 的那條路 ────────────────────────────────────────────────
#
# 供應商與模型都從環境變數來。實測（8 句客服文字，用臺羅音節合法率評分）：
#
#   gemini-3.7-flash        100.0%  4.43s  每次 0.134 元
#   gpt-5.5 / none           99.3%  2.41s  每次 0.116 元   ← 快、準、又最便宜
#   gpt-5.4 / none           93.6%  2.00s  每次 0.058 元
#   gpt-5.6-luna / none      81.1%  2.06s  每次 0.005 元
#   gemini-3.5-flash-lite  40–89%   1.1s   每次 0.017 元
#
# OpenAI 這邊**一定要把推理關掉**（reasoning_effort=none）。開了一律變差：
# luna 從 91% 掉到 80% 還慢五倍，gpt-5-nano 更是掉到 9.3%——它想太多反而編造。
# Gemini 相反，它的思考正是品質來源。兩家的推理對這個任務作用相反。
# 注意新版把參數值從 minimal 改名成 none，舊值會回 400。
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_REASONING = os.environ.get("LLM_REASONING", "none")

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
        path = REPO / "workspace/chhoetaigi/ChhoeTaigi_TaihoaSoanntengTuichiautian.csv"
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
            with open(REPO / "workspace/moe_sutian/multi_readings.json",
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


# 完全不查表，整句交給 Gemini。跟 hanji_llm 的差別是後者只補查表的缺口、
# 表已經換好的詞一個字都不准動；這裡是把整句的判斷都交出去。
_DIRECT_HANJI_PROMPT = """把下面這句華語改寫成**臺灣台語的漢字書寫**。

規則：
1. 輸出仍然是漢字，不要羅馬拼音、不要注音。
2. 用台語實際會講的詞，不是把華語逐字照抄。例如
   東西→物件、明天→明仔載、地上→塗跤、總共→攏總、他→伊、把→共、了→矣。
3. 用教育部推薦用字。沒有推薦用字就用最通行的寫法。
4. 標點、數字、英文原樣保留。
5. 不要解釋、不要加註、不要用引號包起來。只輸出改寫後的那一句。

句子：{s}"""

# 台羅的格式要寫死講清楚，不然它會回傳帶調符的 bîn-á-tsài，而 v7 學的是數字調號。
_DIRECT_TAILO_PROMPT = """把下面這句華語翻成**臺灣台語**，並用**臺羅數字調號**寫出來。

格式規定（非常重要，請嚴格遵守）：
1. **聲調用數字 1–8 寫在每個音節後面**，不要用調符。
   正確：bin5-a2-tsai3　　錯誤：bîn-á-tsài
2. **第 1 聲和第 4 聲也要寫出數字**，不可以省略。
   正確：thinn1-khi3、tsit4　　錯誤：thinn-khi3、tsit
3. 一個詞裡的音節用連字號 - 連起來，詞與詞之間用一個空格。
   例：Bin5-a2-tsai3 thinn1-khi3 tsin1 ho2
4. 全部小寫，只有整句的第一個字母大寫。
5. 標點用半形（, . ? !），標點後面留一個空格。
6. 用台語實際會講的詞，不是華語逐字直譯。
7. 不要附漢字、不要解釋、不要用引號包起來。只輸出臺羅那一行。

參考（這就是要的樣子）：
  華語：明天天氣真好，我們一起去市場買東西。
  臺羅：Bin5-a2-tsai3 thinn1-khi3 tsin1 ho2, lan2 tso3-hue2 khi3 tshi7-tiunn5 be2 mih8-kiann7.

句子：{s}"""


# ── 持久化快取 ────────────────────────────────────────────────────────
#
# 原本只有記憶體字典，重啟就沒了——等於每次重啟都重付一次錢。而客服的回覆
# 多半是樣板，重複率高，快取才是這件事真正的成本控制。
#
# key 包含供應商與模型：換了模型不該拿到舊模型的結果。
_DIRECT_CACHE_PATH = Path(os.environ.get("TAIGI_DATA", "/data/taigi")) / "llm_direct.json"
_direct_cache: dict[str, str] | None = None
_direct_lock = threading.Lock()


def _cache_load() -> dict[str, str]:
    global _direct_cache
    if _direct_cache is None:
        try:
            _direct_cache = json.loads(_DIRECT_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _direct_cache = {}
    return _direct_cache


def _cache_put(key: str, value: str) -> None:
    with _direct_lock:
        cache = _cache_load()
        cache[key] = value
        try:
            _DIRECT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _DIRECT_CACHE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                           encoding="utf-8")
            tmp.replace(_DIRECT_CACHE_PATH)     # 換名是原子的，寫到一半斷電不會壞檔
        except Exception:
            pass                                # 寫不進去就只留在記憶體，不影響合成


# ── 臺羅的正規化與驗證 ─────────────────────────────────────────────────
#
# 模型回傳的不合法音節裡，將近一半是**機械可修**的：
#   去掉調符   mn̄g7 → mng7、lio̍k8 → liok8、thóng2 → thong2   （7/29）
#   POJ → 臺羅 koan1 → kuan1、goa2 → gua2、cha5 → tsa5        （7/29）
# 剩下的是真的拼錯，救不回來——那時整句退回查表，不要讓模型唸不存在的音。
_POJ_TO_TAILO = [("chh", "tsh"), ("ch", "ts"), ("oa", "ua"), ("oe", "ue"),
                 ("eng", "ing"), ("ek", "ik"), ("o͘", "oo"), ("ou", "oo"),
                 ("ⁿ", "nn")]
_syllables: set[str] | None = None


def syllables() -> set[str]:
    """合法的臺羅音節。從 ChhoeTaigi 的讀音欄拆出來，不是手寫的清單。"""
    global _syllables
    if _syllables is None:
        try:
            from taigi_annotate import lexicon
            out: set[str] = set()
            for readings in lexicon().values():
                for r in readings:
                    for piece in re.split(r"[-\s]+", r):
                        if piece:
                            out.add(piece.strip().lower())
            _syllables = out
        except Exception:
            _syllables = set()
    return _syllables


def _strip_marks(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def normalize_tailo(text: str) -> str:
    """把調符與白話字拼法轉成臺羅數字調號。逐音節做，避免動到標點。"""
    def fix(m):
        syl = m.group(0)
        low = _strip_marks(syl).lower()
        if low in syllables():
            return low
        conv = low
        for a, b in _POJ_TO_TAILO:
            conv = conv.replace(a, b)
        return conv if conv in syllables() else low

    out = re.sub(r"[A-Za-z\u00c0-\u024f\u0300-\u036f͘ⁿ]+\d?", fix, text)
    # 逐音節那一步會把全部轉小寫，句首大寫要補回來——v7 的訓練資料是
    # 「Bin5-a2-tsai3 ...」這個形態，見 taigi_annotate._as_trained。
    return re.sub(r"^([a-z])", lambda m: m.group(1).upper(), out.strip())


def invalid_syllables(text: str) -> list[str]:
    known = syllables()
    if not known:
        return []
    toks = [t for t in re.split(r"[-\s,.!?;:、，。？！]+", text.lower()) if t]
    return [t for t in toks if t not in known]


def _openai_text(prompt: str) -> str:
    """問 OpenAI 要一行純文字。"""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("no OPENAI_API_KEY")
    payload = {"model": LLM_MODEL or "gpt-5.5",
               "messages": [{"role": "user", "content": prompt}]}
    if LLM_REASONING:
        # none 是「不要推理」。這個任務開了推理一律變差，見檔頭的實測表。
        payload["reasoning_effort"] = LLM_REASONING
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", body,
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.load(r)
    return raw["choices"][0]["message"]["content"].strip()


def _llm_text(prompt: str) -> str:
    """照 LLM_PROVIDER 決定問誰。"""
    return (_openai_text(prompt) if LLM_PROVIDER == "openai"
            else _gemini_text(prompt))


def _gemini_text(prompt: str) -> str:
    """問 Gemini 要一行純文字。失敗就丟例外，由呼叫端決定怎麼退回。"""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("no GEMINI_API_KEY")
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        # temperature 0：同一句問兩次要拿到同一個答案，否則 A/B 沒有意義
        "generationConfig": {"temperature": 0},
    }).encode()
    url = _GEMINI_URL.format(
        model=LLM_MODEL or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        key=key)
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = json.load(r)
    return raw["candidates"][0]["content"]["parts"][0]["text"].strip()


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


def _direct_fallback(text: str, mode: str, table_version, why: str):
    """LLM 這條路不能用時，退到查表；查表也不行才退回原字。

    退到查表而不是原字，是因為查表又快又安全——它的弱點是涵蓋率，不是正確性。
    直接退回原字等於把已經能換的那些也丟掉。
    """
    base = "hanji" if mode == "hanji_direct" else "corpus"
    try:
        out, note, changed = apply(text, base, table_version)
    except Exception:
        return text, f"{mode}-退回原字（{why}）", []
    if out.strip() == text.strip():
        return text, f"{mode}-退回原字（{why}；查表也換不動）", []
    return out, f"{mode}-退回{base}（{why}）", changed


def apply(text: str, mode: str, table_version: str | None = None,
          extra: dict[str, str] | None = None) -> tuple[str, str, list]:
    """回傳（送進模型的文字, 給人看的說明）。任何失敗都退回原字。"""
    if mode == "none":
        return text, "none", []

    # 完全不查表，整句交給 LLM。
    #
    # 退回是分層的，因為「退到原字」太可惜——查表只要 7 毫秒而且安全，
    # 它的問題是涵蓋率不是正確性。所以順序是：
    #   LLM 通過驗證 → 用它
    #   沒通過       → 退回查表（hanji / corpus）
    #   查表也不行   → 原字送出
    if mode in ("hanji_direct", "tailo_direct"):
        model_id = LLM_MODEL or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
        tag = f"{LLM_PROVIDER}/{model_id}"
        key = f"{LLM_PROVIDER}|{model_id}|{mode}|{text}"
        cache = _cache_load()
        hit = key in cache
        if hit:
            out = cache[key]
        else:
            prompt = (_DIRECT_HANJI_PROMPT if mode == "hanji_direct"
                      else _DIRECT_TAILO_PROMPT)
            try:
                out = _llm_text(prompt.format(s=text))
            except Exception as e:
                return _direct_fallback(text, mode, table_version,
                                        f"{tag} 呼叫失敗:{type(e).__name__}")
            out = out.strip().strip('"「」『』')
            lines = [x.strip() for x in out.splitlines() if x.strip()]
            if lines:
                out = lines[-1]
            for pre in ("臺羅：", "台羅：", "臺語：", "台語：", "漢字："):
                if out.startswith(pre):
                    out = out[len(pre):].strip()
            _cache_put(key, out)
        if not out:
            return _direct_fallback(text, mode, table_version, f"{tag} 回傳空的")

        src = "快取" if hit else tag
        if mode == "tailo_direct":
            if any("\u4e00" <= c <= "\u9fff" for c in out):
                return _direct_fallback(text, mode, table_version, f"{src} 夾雜漢字")
            # 先修可修的：調符與白話字拼法佔了不合法音節的一半
            fixed = normalize_tailo(out)
            bad = invalid_syllables(fixed)
            if bad:
                return _direct_fallback(
                    text, mode, table_version,
                    f"{src} 有 {len(bad)} 個不合法音節[{'/'.join(bad[:3])}]")
            note = f"{mode}:{src}"
            if fixed != out:
                note += "（已正規化）"
            return fixed, note, []
        return out, f"{mode}:{src}", []

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
