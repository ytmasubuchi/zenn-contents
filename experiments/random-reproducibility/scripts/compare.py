"""Diff utility: walk two result JSON files, collect every leaf key that
looks like a hash/hex/bool digest (i.e. anything the experiments use as a
bit-exact fingerprint), and report which ones matched vs diverged.

Usage: python compare.py <label> <file_a.json> <file_b.json> [--out out.json]
"""
import argparse
import json
import sys


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "metadata":
                continue  # metadata differs by definition (hostname etc.)
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.file_a) as f:
        a = json.load(f)
    with open(args.file_b) as f:
        b = json.load(f)

    flat_a = flatten(a)
    flat_b = flatten(b)

    keys = sorted(set(flat_a) | set(flat_b))
    matches, mismatches, only_a, only_b = [], [], [], []
    for k in keys:
        if k not in flat_a:
            only_b.append(k)
        elif k not in flat_b:
            only_a.append(k)
        elif flat_a[k] == flat_b[k]:
            matches.append(k)
        else:
            mismatches.append({"key": k, "a": flat_a[k], "b": flat_b[k]})

    report = {
        "label": args.label,
        "file_a": args.file_a,
        "file_b": args.file_b,
        "metadata_a": a.get("metadata"),
        "metadata_b": b.get("metadata"),
        "n_matches": len(matches),
        "n_mismatches": len(mismatches),
        "matches": matches,
        "mismatches": mismatches,
        "only_in_a": only_a,
        "only_in_b": only_b,
    }

    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
