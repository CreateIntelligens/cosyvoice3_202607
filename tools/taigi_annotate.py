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
TAIGI_WORDS = ROOT / "workspace/moe_sutian/taigi_words.json"

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

    lex = {w: [k for k, _ in sorted(v.items(), key=lambda kv: -kv[1])]
           for w, v in counts.items()}

    # 台華線頂是「詞彙」對照典，沒有單字條目——「十」「三」「沒」全查不到，
    # 整句換不掉就整句退回，覆蓋率被這些常用字卡死。教育部辭典有，補進來。
    import re as _re
    from tailo import to_numeric
    moe = pd.read_excel(KAUTIAN, engine="odf").dropna(subset=["漢字", "羅馬字"])
    for han, roman in zip(moe.漢字, moe.羅馬字):
        han = _re.sub(r"【.*?】", "", str(han)).strip()
        num = to_numeric(_re.sub(r"【.*?】", "", str(roman)).strip())
        if not han or not num or "/" in num or not HAN.search(han):
            continue
        if len(han) > MAX_WORD or not _looks_clean(num):
            continue
        lex.setdefault(han, [])
        if num not in lex[han]:
            lex[han].append(num)
    return lex


@lru_cache(maxsize=1)
def lexicon() -> dict[str, list[str]]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    lex = build_lexicon()
    CACHE.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    return lex


@lru_cache(maxsize=1)
def taigi_words() -> frozenset[str]:
    """教育部辭典收錄的台語詞目——也就是「這串漢字在台語裡是一個詞」。

    這是決定要不要標音的訊號。詞在裡面，模型讀漢字就會念對（市場→tshi7-tiunn5），
    標音是多餘的；不但多餘，還可能挑到比較口語的講法（tshi7-a2 市仔）而降低正式度。
    詞不在裡面，這串漢字在台語不成詞，模型只能逐字硬念（東西、放進），非標不可。

    四個聽測結果四項全中：市場/冰箱在裡面（不標），東西/放進不在（要標）。
    """
    if TAIGI_WORDS.exists():
        return frozenset(json.loads(TAIGI_WORDS.read_text(encoding="utf-8")))
    import pandas as pd
    moe = pd.read_excel(KAUTIAN, engine="odf").dropna(subset=["漢字"])
    words = {re.sub(r"【.*?】", "", str(h)).strip() for h in moe.漢字}
    words.discard("")
    TAIGI_WORDS.write_text(json.dumps(sorted(words), ensure_ascii=False), encoding="utf-8")
    return frozenset(words)


NUMERAL = set("零一二三四五六七八九十百千萬兩0123456789")
CLASSIFIER = set("個个隻張條本台間位次件顆杯碗盤床把塊份組份年月日點分秒歲")


def segment(text: str, lex: dict[str, list[str]]):
    """最長詞優先切分。

    逐字查是行不通的：「乾坤」拆開查會得到 kan1（乾燥的乾）+ khun1，
    而整個詞查是 khian5-khun1。破音字要的是詞層級的知識。

    但最長優先會反咬：「六個人」裡的「個人」是辭典詞（個體、個人），於是切成
    六＋個人，唸出來變成 lak8 ko3-jin5。數詞後面的量詞要單獨切開。
    """
    out, i = [], 0
    while i < len(text):
        prev_num = i > 0 and text[i - 1] in NUMERAL
        for n in range(min(MAX_WORD, len(text) - i), 0, -1):
            w = text[i:i + n]
            # 數詞後面接量詞時，量詞不跟後面的字黏成詞
            if n > 1 and prev_num and w[0] in CLASSIFIER:
                continue
            if w in lex:
                out.append((w, lex[w]))
                i += n
                break
        else:
            out.append((text[i], None))
            i += 1
    return out


# 全有全無的單位是「整句」（句號為界），不是子句。以子句為單位的話，
# 「桌上有六個蘋果，我拿了三個。」前半換得掉、後半換不掉，結果同一個句子裡
# 台羅與漢字並存——那正是 1/10 會幻覺的混排形態。
SENTENCE = re.compile(r"[^。！？\n]+[。！？\n]*|[。！？\n]+")


CORPUS_PRIOR = ROOT / "workspace/chhoetaigi/corpus_prior.json"


@lru_cache(maxsize=1)
def corpus_prior() -> dict[str, str]:
    """每個詞在真實台語文本裡最常用的讀音。

    從 naer 的 107,292 句雙寫法學出來的：華語那側切出詞，看台語那側實際寫成
    什麼。3,717 個詞。這比辭典詞頻準得多，因為辭典數的是「有幾個詞條」，
    語料數的是「人實際上怎麼講」。

        辭典詞頻    46.1%
        語料預設    86.6%      ← 零 API 呼叫

    它不看句子，所以「我 vs 阮」這種要看上下文的救不到。天花板 94.2%，
    剩下 5.8% 加上沒學過的詞（9.1%）才需要問 LLM。
    """
    if CORPUS_PRIOR.exists():
        return json.loads(CORPUS_PRIOR.read_text(encoding="utf-8"))
    return {}


def by_corpus(word: str, readings: list[str]) -> str:
    """語料學出來的預設，學不到就退回辭典詞頻。

    預設表裡的讀音來自教育部辭典，**不要求它出現在台華線頂的候選清單裡**。
    那份清單對「台語用哪個詞」有用，對「那個詞怎麼念」不權威——「影響」它只
    收 ing2-hiang2，教育部是 ing2-hiong2，而後者才是對的。
    """
    return corpus_prior().get(word) or readings[0]


def most_frequent(word: str, readings: list[str]) -> str:
    """辭典裡出現最多次的讀音。**這個挑法會錯得很兇。**

        把 → lak8（該是 ka7）      冰箱 → ping1-kui7 冰櫃（該是 ping1-siunn1）
        東西 → mih8（該是 mih8-kiann7）  準備 → tshuan5（該是 tsun2-pi7）

    出現次數跟「這一句該用哪個」沒有關係。真的要選對得看上下文，那是
    taigi_disambiguate.py 的工作——把它的 resolver 當 pick 傳進來。

    留這個當預設是因為要有東西可退，但它產出的是**流利的錯話**：模型會穩定地
    念出來，聽起來很順，講的卻不是那個意思。比幻覺更難察覺。
    """
    return readings[0]


# v7 看過的台羅長這樣：「Gua2 thiann1 li2 leh4 khau3-iau1.」
# 半形標點 5,700 次、全形只有 15 次，而且句首大寫。送全形＋全小寫等於送它
# 沒見過的格式，聽起來會像把詞硬黏在一起——那不是選詞的問題，是格式的問題。
_FULLWIDTH = str.maketrans({"，": ",", "。": ".", "！": "!", "？": "?",
                            "；": ";", "：": ":", "、": ","})


def _as_trained(line: str) -> str:
    line = line.translate(_FULLWIDTH)
    line = re.sub(r"\s+([,.!?;:])", r"\1", line)      # 標點前不留空格
    line = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", line)   # 標點後留一個空格
    return re.sub(r"^([a-z])", lambda m: m.group(1).upper(), line.strip())


def annotate_clause_wise(text: str, *, only_unambiguous: bool = False, pick=by_corpus):
    """整句標，或整句不標——不產生漢字與台羅混在一起的句子。

    量過 10 個 seed：混排 1/10 會幻覺（3.5 秒的句子生出 21.6 秒），純漢字與
    全台羅都是 0/10。漢字夾在台羅中間是唯一會出事的形態，所以寧可整句不標。

    一個子句裡只要有一個詞查不到讀音，整句原樣保留。這讓覆蓋率變低，但低覆蓋率
    是安全的失敗——不標就是現在 v7 的行為，而混排會生出幻覺。

    pick  怎麼從多個讀音裡挑一個。預設用語料學出的讀音（by_corpus，86.6%）。
          要再上去得看句子，那是 taigi_disambiguate 的工作。
    """
    lex = lexicon()
    out, changed = [], []
    for clause in SENTENCE.findall(text):
        if not clause.strip() or not re.search(r"[一-鿿]", clause):
            out.append(clause)
            continue
        words = segment(clause, lex)
        # 整句要換就每個詞都得有讀音，包括模型本來念得對的那些
        resolved = []
        for word, readings in words:
            if not word.strip():
                continue
            if readings is None:
                # 標點原樣留著，它不是查不到讀音的字
                if not re.search(r"[一-鿿A-Za-z0-9]", word):
                    resolved.append((word, word))
                    continue
                resolved = None
                break
            if only_unambiguous and len(readings) > 1:
                resolved = None
                break
            resolved.append((word, pick(word, readings)))
        if resolved is None:
            out.append(clause)
            continue
        out.append(_as_trained(" ".join(k for _, k in resolved)))
        changed.extend((w, k, len(lex.get(w, [k]))) for w, k in resolved
                       if re.search(r"[一-鿿]", w))
    text = re.sub(r" +([，。！？；：、])", r"\1", "".join(out))
    return text.strip(), changed


def annotate(text: str, *, only_unambiguous: bool = True,
             wordlist: set[str] | None = None, skip_taigi_words: bool = True):
    """逐詞標音。回傳（標好的文字, 改了哪些詞）。

    **會產生混排，而混排 1/10 會幻覺。** 一般用途請用 annotate_clause_wise。
    這支留著是因為指定 wordlist 手工標單一個詞時仍然有用（研究、對照實驗）。

    標音一律「換掉原字」。試過並列的寫法——「乾坤(khian5-khun1)」模型會念兩遍，
    先念漢字再念羅馬字，等於沒覆寫到還多講一次。要覆寫就得把字換掉。

    only_unambiguous  辭典給出多種讀音時不標。挑錯讀音比不標更糟——
                      不標至少還有模型自己的判斷，標錯是直接指定念錯。
                      多義詞要標的話交給 taigi_disambiguate.py
    wordlist          只標這些詞。給 None 表示標所有查得到的
    skip_taigi_words  台語本來就有的詞不標，交給模型自己念（見 taigi_words）
    """
    lex = lexicon()
    native = taigi_words() if skip_taigi_words else frozenset()
    pieces, changed = [], []
    for word, readings in segment(text, lex):
        if readings is None:
            pieces.append(word)
            continue
        if wordlist is not None and word not in wordlist:
            pieces.append(word)
            continue
        if word in native:
            pieces.append(word)
            continue
        if only_unambiguous and len(readings) > 1:
            pieces.append(word)
            continue
        kip = readings[0]
        pieces.append(f" {kip} ")
        changed.append((word, kip, len(readings)))
    # 換字會在中文之間留下多餘空白，收乾淨
    text = re.sub(r" {2,}", " ", "".join(pieces))
    return re.sub(r"\s+([，。！？；：、])", r"\1", text).strip(), changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="要標音的華語文字；省略則讀 stdin")
    ap.add_argument("--all", action="store_true",
                    help="多義詞也標（取最常見的讀音）。預設只標唯一讀音的詞")
    ap.add_argument("--words", help="只標這些詞，逗號分隔")
    ap.add_argument("--annotate-native", action="store_true",
                    help="連台語本來就有的詞也標。預設不標，那些模型自己念得對")
    ap.add_argument("--show", action="store_true", help="列出改了哪些詞")
    ap.add_argument("--strict", action="store_true",
                    help="整句模式下，子句裡有多義詞就整句不標。預設取最常見的讀音")
    ap.add_argument("--per-word", action="store_true",
                    help="逐詞標，會產生混排。混排 1/10 會幻覺，只在做對照實驗時用")
    a = ap.parse_args()

    text = a.text if a.text else sys.stdin.read().strip()
    if a.per_word or a.words:
        out, changed = annotate(
            text,
            only_unambiguous=not a.all,
            wordlist=set(a.words.split(",")) if a.words else None,
            skip_taigi_words=not a.annotate_native,
        )
    else:
        out, changed = annotate_clause_wise(text, only_unambiguous=a.strict)
        prior = corpus_prior()
        guessed = [w for w, _, n in changed if n > 1 and w not in prior]
        if guessed:
            print(f"  警告：{len(guessed)} 個詞語料裡沒學過，退回辭典詞頻（那個挑法會錯）"
                  f"：{'、'.join(guessed[:6])}", file=sys.stderr)
    print(out)
    if a.show:
        for w, k, n in changed:
            note = "" if n == 1 else f"  ← 辭典有 {n} 種讀音，取最常見的"
            print(f"  {w} → {k}{note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
