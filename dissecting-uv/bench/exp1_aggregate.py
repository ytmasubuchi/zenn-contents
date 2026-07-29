"""実験1: 生ログ(jsonl)を読み込み、中央値集計をJSON/Markdownで出力する"""
import json
import statistics

FILES = {
    "pip": "/bench/results/exp1_pip_raw.jsonl",
    "poetry": "/bench/results/exp1_poetry_raw.jsonl",
    "uv": "/bench/results/exp1_uv_raw.jsonl",
}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize():
    summary = {}
    for tool, path in FILES.items():
        rows = load(path)
        for cond in ("cold", "warm"):
            sub = [r for r in rows if r["condition"] == cond]
            if not sub:
                continue
            totals = [r["total_sec"] for r in sub]
            key = f"{tool}_{cond}"
            entry = {
                "tool": tool,
                "condition": cond,
                "n": len(sub),
                "all_total_sec": totals,
                "median_total_sec": round(statistics.median(totals), 3),
            }
            if "lock_sec" in sub[0]:
                locks = [r["lock_sec"] for r in sub]
                installs = [r["install_sec"] for r in sub]
                entry["all_lock_sec"] = locks
                entry["median_lock_sec"] = round(statistics.median(locks), 3)
                entry["all_install_sec"] = installs
                entry["median_install_sec"] = round(statistics.median(installs), 3)
            summary[key] = entry
    return summary


if __name__ == "__main__":
    summary = summarize()
    out_path = "/bench/results/exp1_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    for key, e in summary.items():
        print(f"{key}: median_total={e['median_total_sec']}s n={e['n']} all={e['all_total_sec']}")
    print(f"wrote {out_path}")
