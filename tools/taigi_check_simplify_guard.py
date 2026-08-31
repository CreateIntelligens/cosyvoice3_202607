#!/usr/bin/env python3
"""檢查繁簡轉換不會改壞台文對照表換出來的詞。

為什麼需要這個檢查
------------------
繁簡轉換原本是為華語輸入設計的：Qwen 的詞彙表偏簡體，罕見繁體字沒有共用字形
就會碎成 UTF-8 位元組，模型從未把它們當成一個字看待，於是唸錯甚至整段漏掉。
實測紅樓夢那段 71 字就掉了「成扇墜一般，甚屬」，日誌標出 `仍是碎片[瑩墜鐫攜纓]`。

但同一個轉換套在台語漢字上會壞事：「連紲」變「连绁」，紲 lian5-sua3 變成
绁 xiè，讀音就跟著錯。所以原本的做法是二選一——走台文查表就整段不轉。

那是為了避開 98 個字，放棄了另外 2192 個。現在改成兩件事都做，靠兩層保護：

  keep_words  這一次查表**實際換出來**的詞，逐字保留。精準，不會誤傷——
              「塗」在華語語境裡照樣可以轉，只有查表產出的「塗跤」不動。
  protect     整張表用到的字。安全網，接住使用者自己打的台文漢字。

這個檔案就是那個保證的可執行版本：拿對照表自己當測資，每一條規則造一句話，
走完整條路徑，然後檢查產出的台文詞還在不在。改表、改程式、換 tokenizer 之後
都該重跑——這種退化不會報錯，只會讓人聽到怪音卻找不到原因。

用法
----
    docker exec cosyvoice3-testbench python /opt/CosyVoice/tools/taigi_check_simplify_guard.py

    --sample N   抽幾條來測（預設 400；0 表示全部）
    --version V  用哪個版本的對照表（預設 v5，跟 /v1 一致）

離開碼 0 表示全部通過。
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "docker" / "testbench" / "app"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=400, help="抽幾條，0 = 全部")
    ap.add_argument("--version", default="v5", help="對照表版本")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import annotate_layer
    import server
    import taigi_store

    table = taigi_store.merged(args.version)
    pairs = [(k, v) for k, v in table.items() if v != k]
    if not pairs:
        print(f"對照表 {args.version} 是空的，沒東西可以檢查", file=sys.stderr)
        return 2

    random.seed(args.seed)
    sample = pairs if args.sample <= 0 else random.sample(
        pairs, min(args.sample, len(pairs)))

    print(f"對照表 {args.version}：{len(pairs)} 條可換的詞，抽 {len(sample)} 條")

    modes = ("none", "safe", "full")
    broken: dict[str, list] = {m: [] for m in modes}
    unguarded: list = []
    triggered = 0

    for zh, tai in sample:
        # 造一個最普通的語境。查表是照分詞結果比對的，所以不能只丟一個孤立的詞。
        text = f"我今天{zh}好幾次。"
        out, _note, changed = annotate_layer.apply(text, "hanji", args.version)
        if tai not in out:
            continue          # 這一條沒被分詞切到，不算數
        triggered += 1
        keep = [g for _w, g in (changed or [])]
        for mode in modes:
            got, _ = server.to_simplified(out, mode, protect=True, keep_words=keep)
            if tai not in got:
                broken[mode].append((zh, tai, got))
        # 對照組：不保護的話會壞幾條
        got, _ = server.to_simplified(out, "safe")
        if tai not in got:
            unguarded.append((zh, tai))

    print(f"實際被分詞觸發的：{triggered} 條\n")
    failed = False
    for mode in modes:
        bad = broken[mode]
        mark = "OK" if not bad else "★ 失敗"
        print(f"  simplify={mode:<5s} 被改壞 {len(bad):3d} 條   {mark}")
        for zh, tai, got in bad[:5]:
            print(f"      {zh} → {tai}，轉完變成 {got}")
        failed = failed or bool(bad)

    print(f"\n  對照（關掉保護）：simplify=safe 被改壞 {len(unguarded)} 條")
    if unguarded:
        print("      " + "、".join(f"{z}→{t}" for z, t in unguarded[:6]))

    if failed:
        print("\n保護失效。繁簡轉換正在改壞台文的讀音——先把 simplify 設回 none，"
              "再看是 keep_words 沒傳到，還是 taigi_chars 沒涵蓋。")
        return 1
    print("\n通過：台文查表的產出在三種繁簡模式下都完整保留。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
