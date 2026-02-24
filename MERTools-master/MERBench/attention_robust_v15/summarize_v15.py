#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


FILE_RE = re.compile(
    r"^(?P<split>cv|test1|test2|test3)_.*_f1:(?P<f1>-?\d+(?:\.\d+)?)_acc:(?P<acc>-?\d+(?:\.\d+)?)_val:(?P<val>-?\d+(?:\.\d+)?)_"
)


def parse_result_dir(result_dir: Path):
    metrics = {}
    for file in result_dir.glob("*.npz"):
        m = FILE_RE.match(file.name)
        if not m:
            continue
        split = m.group("split")
        item = {
            "f1": float(m.group("f1")),
            "acc": float(m.group("acc")),
            "val": float(m.group("val")),
            "file": str(file),
            "_mtime": file.stat().st_mtime,
        }
        # If the same split exists multiple times in one folder (reruns),
        # keep the latest file by mtime.
        old = metrics.get(split)
        if old is None or item["_mtime"] >= old["_mtime"]:
            metrics[split] = item

    for split in list(metrics.keys()):
        metrics[split].pop("_mtime", None)
    return metrics


def compute_table_metrics(metrics):
    train_waf = metrics["cv"]["f1"] * 100.0
    train_mse = metrics["cv"]["val"]
    multi_waf = metrics["test1"]["f1"] * 100.0
    multi_mse = metrics["test1"]["val"]
    noise_waf = metrics["test2"]["f1"] * 100.0
    noise_mse = metrics["test2"]["val"]
    semi_waf = metrics["test3"]["f1"] * 100.0

    avg_waf = (train_waf + multi_waf + noise_waf + semi_waf) / 4.0
    avg_mse = (train_mse + multi_mse + noise_mse) / 3.0

    # Equivalent to f1 - 0.25*mse, but on percentage scale.
    score_multi = multi_waf - 25.0 * multi_mse
    score_balanced = avg_waf - 25.0 * avg_mse

    return {
        "train_waf": train_waf,
        "train_mse": train_mse,
        "multi_waf": multi_waf,
        "multi_mse": multi_mse,
        "noise_waf": noise_waf,
        "noise_mse": noise_mse,
        "semi_waf": semi_waf,
        "avg_waf": avg_waf,
        "avg_mse": avg_mse,
        "score_multi": score_multi,
        "score_balanced": score_balanced,
    }


def mean_std(values):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def collect_runs(root: Path, tag_prefix: str = None, tag_regex=None):
    runs = []
    for result_dir in root.glob("**/result"):
        if not result_dir.is_dir():
            continue
        parsed = parse_result_dir(result_dir)
        if not {"cv", "test1", "test2", "test3"}.issubset(parsed.keys()):
            continue

        parts = result_dir.parts
        if "results" in parts:
            idx = parts.index("results")
            run_tag = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
            seed_tag = parts[idx + 2] if idx + 2 < len(parts) else "unknown_seed"
        else:
            run_tag = "unknown"
            seed_tag = "unknown_seed"

        row = {
            "run_tag": run_tag,
            "seed_tag": seed_tag,
            "result_dir": str(result_dir),
        }
        row.update(compute_table_metrics(parsed))
        if tag_prefix and not row["run_tag"].startswith(tag_prefix):
            continue
        if tag_regex and not tag_regex.search(row["run_tag"]):
            continue
        runs.append(row)
    return runs


def aggregate_by_tag(runs):
    grouped = defaultdict(list)
    for row in runs:
        grouped[row["run_tag"]].append(row)

    metric_keys = [
        "train_waf",
        "train_mse",
        "multi_waf",
        "multi_mse",
        "noise_waf",
        "noise_mse",
        "semi_waf",
        "avg_waf",
        "avg_mse",
        "score_multi",
        "score_balanced",
    ]
    summary_rows = []

    for run_tag, items in grouped.items():
        out = {
            "run_tag": run_tag,
            "runs_ok": len(items),
        }
        for key in metric_keys:
            vals = [x[key] for x in items]
            m, s = mean_std(vals)
            out[f"{key}_mean"] = m
            out[f"{key}_std"] = s
        summary_rows.append(out)

    summary_rows.sort(
        key=lambda x: (
            x["multi_waf_mean"],
            x["avg_waf_mean"],
            -x["avg_mse_mean"],
        ),
        reverse=True,
    )
    return summary_rows


def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize v15 runs into Train&Val / MER-MULTI / MER-NOISE / MER-SEMI table format."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("attention_robust_v15/outputs/results"),
        help="root directory containing run outputs",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("attention_robust_v15/outputs/summary/summary.csv"),
        help="output CSV path",
    )
    parser.add_argument("--topk", type=int, default=10, help="show top-k rows in terminal")
    parser.add_argument(
        "--tag_prefix",
        type=str,
        default=None,
        help="only include run_tag starting with this prefix",
    )
    parser.add_argument(
        "--tag_regex",
        type=str,
        default=None,
        help="only include run_tag matching this regex",
    )
    args = parser.parse_args()

    tag_regex = re.compile(args.tag_regex) if args.tag_regex else None
    runs = collect_runs(args.root, tag_prefix=args.tag_prefix, tag_regex=tag_regex)
    summary = aggregate_by_tag(runs)
    write_csv(summary, args.out_csv)

    print(f"runs_parsed={len(runs)}")
    print(f"summary_rows={len(summary)}")
    print(f"csv={args.out_csv}")

    topk = min(args.topk, len(summary))
    for i in range(topk):
        row = summary[i]
        print(
            f"[{i+1}] {row['run_tag']} "
            f"n={row['runs_ok']} "
            f"multi={row['multi_waf_mean']:.2f}±{row['multi_waf_std']:.2f} "
            f"noise={row['noise_waf_mean']:.2f}±{row['noise_waf_std']:.2f} "
            f"semi={row['semi_waf_mean']:.2f}±{row['semi_waf_std']:.2f} "
            f"avg={row['avg_waf_mean']:.2f}±{row['avg_waf_std']:.2f} "
            f"avg_mse={row['avg_mse_mean']:.4f}±{row['avg_mse_std']:.4f}"
        )


if __name__ == "__main__":
    main()
