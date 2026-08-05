#!/usr/bin/env python3
"""Generate every (model, text, seed) combination through the testbench API.

Listening to one clip at a time through the web UI does not scale: comparing six
checkpoints over eight passages with three seeds each is 144 syntheses, and every
checkpoint switch costs another 20 s of model loading.  This walks the grid with
the model on the outer loop so each checkpoint is loaded exactly once, and writes
the clips into per-passage folders so the same sentence from different models sits
side by side.

Seeds matter as much as the checkpoints.  A model that only sounds right on one
lucky seed is worse than one that is consistently decent, so the same seeds are
reused across models and the per-model spread is what to listen for.

    python batch_eval.py --models llm_v2_best llm_v3a_best llm_v3e_naer_best
    python batch_eval.py --texts 2 4 --seeds 1942676902 --models llm_v2_best
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys
import time

import requests


# 每一段專門測一個弱點；長度壓在 60 字以內，因為這張卡與其他服務共用，
# 長輸入的 KV cache 會把僅存的餘裕吃光。
PASSAGES: dict[str, tuple[str, str]] = {
    "1_news_social": (
        "新北市新莊天空垃圾山已啟動二階段清除，昨日首日清除運出逾五公噸垃圾。",
        "垃圾、天空——已知會念錯的詞",
    ),
    "2_news_crime": (
        "警方調查後發現，兇嫌犯案後隨即駕車逃逸，目前已鎖定身分並發布通緝。",
        "兇嫌、犯案、逃逸、通緝——在 tai8 與 naer 都是 0 次的新聞詞",
    ),
    "3_news_finance": (
        "台積電第三季營收較去年同期成長百分之三十七，每股盈餘達十二點五元。",
        "數字、百分比、單位——台語數字念法與華語差異大",
    ),
    "4_narrative_epic": (
        "東漢末年，天下大亂，群雄並起。曹操挾天子以令諸侯，劉備三顧茅廬請出諸葛亮。",
        "文言詞與四字成語——最難的一類",
    ),
    "5_narrative_plain": (
        "從前有一個老阿伯，住在山腳下，每天早上都會挑著扁擔去市場賣菜。",
        "敘事體但用詞口語——介於戲劇與朗讀之間",
    ),
    "6_daily": (
        "明天早上九點，我們一起去市場買東西，順便看看有沒有便宜的青菜。",
        "日常對話——模型最強的一類，當天花板參考",
    ),
    "7_mixed_en": (
        "這家店專營日版寶可夢 PTCG 代購，也提供美國和日本的鑑定服務。",
        "英文縮寫與外來語",
    ),
    "8_proper_noun": (
        "他從彰化搭高鐵到左營，再轉台鐵到屏東潮州，拜訪住在那裡的林先生。",
        "地名人名——台語念法與華語差異大",
    ),

    # 9–14 是為了比較 v2（naer 用台文）與 v4（naer 轉華語）而加的。兩種書寫系統
    # 的差異幾乎都落在虛詞上——不/毋、在/佇、都/攏、要/欲、沒有/無——而實詞
    # （垃圾、天氣、清除）兩邊寫法相同。所以差異該出現在虛詞密集的句子，
    # 而不是以實詞為主的新聞稿。第 14 段是對照組，兩版應該一樣。
    "9_function_neg": (
        "我不知道他在哪裡，也沒有人告訴我，這件事情不能再拖下去了。",
        "不、在、沒有——台文寫作毋、佇、無",
    ),
    "10_function_query": (
        "你要不要跟我們一起去？什麼時候出發比較好？大家都還沒決定。",
        "要、什麼、比較、都——台文寫作欲、啥物、較、攏",
    ),
    "11_function_deixis": (
        "這個給你，那個放在桌子上面就好，其他的都不用拿過來。",
        "這個、那個、都——台文寫作這个、彼个、攏",
    ),
    "12_naer_weather": (
        "明天各地大多是晴朗的天氣，中午過後山區有短暫陣雨，出門記得帶傘。",
        "天氣預報——naer 的主要內容，轉換後應最有感",
    ),
    "13_naer_announce": (
        "請一百二十三號的旅客到三號櫃台辦理手續，謝謝您的配合。",
        "服務廣播——naer 的另一大宗，含數字",
    ),
    "14_content_control": (
        "台灣的高速公路系統連接北部和南部的主要城市，全長超過三百公里。",
        "對照組：幾乎全是實詞，兩版寫法相同，預期沒有差異",
    ),
}

DEFAULT_SEEDS = [1942676902, 20260803, 777]
DEFAULT_MODELS = ["llm_v2_best", "llm_v3a_best", "llm_v3e_naer_best"]


def synthesise(api: str, out: Path, *, text: str, llm: str, flow: str, hift: str,
               prompt: str, prompt_wav: Path | None, seed: int, speed: float,
               timeout: int, simplify: str = "none") -> tuple[bool, str]:
    data = {
        "text": text, "llm": llm, "flow": flow, "hift": hift,
        "seed": str(seed), "speed": str(speed), "text_frontend": "false",
        "simplify": simplify,
    }
    files = None
    if prompt_wav is not None:
        # 每次請求都要重新開檔：伺服器端會把上傳內容存成暫存檔再讀三次
        files = {"prompt_wav": (prompt_wav.name, prompt_wav.read_bytes())}
    else:
        data["prompt_name"] = prompt
    try:
        r = requests.post(f"{api}/api/tts", data=data, files=files, timeout=timeout)
    except requests.RequestException as error:
        return False, f"連線失敗: {error}"
    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("detail", "")
        except Exception:
            detail = r.text[:160]
        return False, f"HTTP {r.status_code}: {detail}"
    out.write_bytes(r.content)
    return True, r.headers.get("X-Seed", str(seed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:50010")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="LLM checkpoint 名稱")
    parser.add_argument("--flow", default="official")
    parser.add_argument("--hift", default="official")
    parser.add_argument("--prompt", default="holdout_male_drama1_680.wav", help="預設清單中的音色")
    parser.add_argument("--prompt-wav", default="", help="改用自己的音檔（1–30 秒，建議 3–10 秒）")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--texts", nargs="+", default=[], help="只跑這幾段（給編號，例如 2 4）")
    # 繁體字在 Qwen 的詞彙表裡沒有共用字形就退化成 UTF-8 位元組，模型從未把它們
    # 當成一個字。實測（official 模型、合成華語）：不轉時「兇」「颱」「廬」都念錯，
    # 只修碎片字就全部念對；整句轉簡沒有更好，反而會動到「後」「麵」「乾」而念錯。
    # 交給伺服器做，這裡只轉達模式，才不會和試聽台的行為分岔。
    parser.add_argument("--simplify", choices=("none", "safe", "full"), default="none",
                        help="繁體字轉簡體字形：safe 只修會碎片的，full 整句轉")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--list", action="store_true", help="列出所有測試文本後結束")
    args = parser.parse_args()

    if args.list:
        for key, (text, probes) in PASSAGES.items():
            print(f"{key}\n  {text}\n  測：{probes}\n")
        return 0

    chosen = PASSAGES
    if args.texts:
        wanted = {t.split("_")[0] for t in args.texts}
        chosen = {k: v for k, v in PASSAGES.items() if k.split("_")[0] in wanted}
        if not chosen:
            print(f"沒有符合的文本編號: {args.texts}", file=sys.stderr)
            return 2

    prompt_wav = Path(args.prompt_wav) if args.prompt_wav else None
    if prompt_wav is not None:
        if not prompt_wav.is_file():
            print(f"找不到音檔: {prompt_wav}", file=sys.stderr)
            return 2
        try:
            import soundfile as sf
            duration = sf.info(str(prompt_wav)).duration
            if not 1.0 <= duration <= 30.0:
                print(f"音檔 {duration:.2f} 秒，超出 1–30 秒的範圍", file=sys.stderr)
                return 2
            if duration > 10.0:
                print(f"提醒：音檔 {duration:.2f} 秒偏長，建議 3–10 秒；"
                      f"過長會多吃顯存，而這張卡與其他服務共用")
        except ImportError:
            pass

    if args.simplify != "none":
        print(f"轉換模式：{args.simplify}（由伺服器套用，日誌會記錄實際換掉幾個字）")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.out) if args.out else Path("eval") / stamp
    root.mkdir(parents=True, exist_ok=True)

    total = len(args.models) * len(chosen) * len(args.seeds)
    print(f"{len(args.models)} 個模型 × {len(chosen)} 段文本 × {len(args.seeds)} 個 seed = {total} 次合成")
    print(f"輸出：{root}\n")

    rows, done, failed = [], 0, 0
    started = time.time()
    # 模型放外層：每個 checkpoint 只載入一次，省下大量重新載入的時間
    for llm in args.models:
        print(f"── {llm}")
        for key, (text, probes) in chosen.items():
            folder = root / key
            folder.mkdir(exist_ok=True)
            for seed in args.seeds:
                done += 1
                tag = "" if args.simplify == "none" else f"_{args.simplify}"
                name = f"{key}__{llm}{tag}__seed{seed}.wav"
                ok, info = synthesise(
                    args.api, folder / name, text=text, llm=llm, flow=args.flow,
                    hift=args.hift, prompt=args.prompt, prompt_wav=prompt_wav,
                    seed=seed, speed=args.speed, timeout=args.timeout,
                    simplify=args.simplify,
                )
                mark = "✓" if ok else "✗"
                print(f"   [{done:>3}/{total}] {mark} {key} seed={seed}" + ("" if ok else f"  {info}"))
                if not ok:
                    failed += 1
                rows.append({
                    "passage": key, "probes": probes, "llm": llm, "flow": args.flow,
                    "hift": args.hift,
                    "prompt": prompt_wav.name if prompt_wav else args.prompt, "seed": seed,
                    "ok": ok, "note": "" if ok else info,
                    "file": str((folder / name).relative_to(root)) if ok else "",
                    "text": text, "simplify": args.simplify,
                })

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 空白的評分表，聽的時候直接填，才不會憑印象比較
    score = root / "scores.csv"
    with score.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["passage", "llm", "seed", "中文字數", "念錯字數", "詞內混雜", "備註"])
        for row in rows:
            if row["ok"]:
                writer.writerow([row["passage"], row["llm"], row["seed"], "", "", "", ""])

    elapsed = time.time() - started
    print(f"\n完成 {done - failed}/{total}，失敗 {failed}，耗時 {elapsed / 60:.1f} 分鐘")
    print(f"  音檔    {root}")
    print(f"  清單    {manifest}")
    print(f"  評分表  {score}  ← 聽的時候直接填")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
