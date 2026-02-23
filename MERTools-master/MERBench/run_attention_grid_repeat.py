#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


METRIC_RE = re.compile(
    r"cv_.*?_f1:(?P<f1>-?\d+(?:\.\d+)?)_acc:(?P<acc>-?\d+(?:\.\d+)?)_val:(?P<val>-?\d+(?:\.\d+)?)_"
)


def emoval_score(f1, val):
    return f1 - 0.25 * val


def write_hyper_yaml(path, model_name, hidden_dim, dropout, lr, grad_clip):
    content = (
        f"{model_name}:\n"
        f"  hidden_dim: {hidden_dim}\n"
        f"  dropout: {dropout}\n"
        f"  grad_clip: {grad_clip}\n"
        f"  lr: {lr}\n"
    )
    path.write_text(content, encoding="utf-8")


def extract_metrics(log_text):
    last = None
    for match in METRIC_RE.finditer(log_text):
        last = {
            "f1": float(match.group("f1")),
            "acc": float(match.group("acc")),
            "val": float(match.group("val")),
        }
    return last


def run_once(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_text = []
    with log_path.open("w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fout.write(line)
            full_text.append(line)
        ret = proc.wait()
    return ret, "".join(full_text)


def aggregate_results(rows):
    grouped = {}
    for row in rows:
        key = row["candidate_id"]
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, items in grouped.items():
        first = items[0]
        vals = [x["emoval"] for x in items if x["status"] == "ok"]
        f1s = [x["f1"] for x in items if x["status"] == "ok"]
        mses = [x["val"] for x in items if x["status"] == "ok"]
        accs = [x["acc"] for x in items if x["status"] == "ok"]
        if len(vals) == 0:
            mean_emoval = None
            std_emoval = None
            mean_f1 = None
            mean_val = None
            mean_acc = None
        else:
            mean_emoval = statistics.mean(vals)
            std_emoval = statistics.stdev(vals) if len(vals) > 1 else 0.0
            mean_f1 = statistics.mean(f1s)
            mean_val = statistics.mean(mses)
            mean_acc = statistics.mean(accs)
        summary.append(
            {
                "candidate_id": key,
                "hidden_dim": first["hidden_dim"],
                "dropout": first["dropout"],
                "lr": first["lr"],
                "runs_ok": len(vals),
                "runs_total": len(items),
                "mean_emoval": mean_emoval,
                "std_emoval": std_emoval,
                "mean_f1": mean_f1,
                "mean_acc": mean_acc,
                "mean_val": mean_val,
            }
        )

    summary.sort(
        key=lambda x: (-1e9 if x["mean_emoval"] is None else x["mean_emoval"]),
        reverse=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Small-grid hyperparameter tuning + repeated runs for attention model."
    )
    parser.add_argument("--python_bin", type=str, default="/root/miniconda3/bin/python")
    parser.add_argument("--model", type=str, default="attention")
    parser.add_argument("--dataset", type=str, default="MER2023")
    parser.add_argument("--feat_type", type=str, default="utt")
    parser.add_argument("--audio_feature", type=str, default="chinese-hubert-large-UTT")
    parser.add_argument("--text_feature", type=str, default="Baichuan-13B-Base-UTT")
    parser.add_argument("--video_feature", type=str, default="clip-vit-large-patch14-UTT")
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.3, 0.4])
    parser.add_argument("--lrs", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--grad_clip", type=float, default=-1.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--base_seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save_root_base", type=str, default="./saved_grid_repeat")
    parser.add_argument("--output_dir", type=str, default="./logs/attention_grid_repeat")
    parser.add_argument("--max_candidates", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent
    output_dir = (workdir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hparam_dir = output_dir / "hparams"
    runlog_dir = output_dir / "runs"
    hparam_dir.mkdir(parents=True, exist_ok=True)
    runlog_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(itertools.product(args.hidden_dims, args.dropouts, args.lrs))
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    print(f"candidate count: {len(candidates)}")
    print(f"repeats per candidate: {args.repeats}")
    print(f"total runs: {len(candidates) * args.repeats}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_rows = []

    for cand_idx, (hidden_dim, dropout, lr) in enumerate(candidates, start=1):
        candidate_id = f"cand{cand_idx:02d}_hd{hidden_dim}_do{dropout}_lr{lr}"
        hyper_path = hparam_dir / f"{candidate_id}.yaml"
        write_hyper_yaml(
            path=hyper_path,
            model_name=args.model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            lr=lr,
            grad_clip=args.grad_clip,
        )

        for rep in range(1, args.repeats + 1):
            seed = args.base_seed + rep - 1
            save_root = f"{args.save_root_base}/{candidate_id}/rep{rep:02d}"
            run_name = f"{candidate_id}_rep{rep:02d}"
            run_log = runlog_dir / f"{run_name}.log"
            cmd = [
                args.python_bin,
                "-u",
                "main-release.py",
                f"--model={args.model}",
                f"--dataset={args.dataset}",
                f"--feat_type={args.feat_type}",
                f"--audio_feature={args.audio_feature}",
                f"--text_feature={args.text_feature}",
                f"--video_feature={args.video_feature}",
                f"--hyper_path={str(hyper_path)}",
                f"--l2={args.l2}",
                f"--epochs={args.epochs}",
                f"--batch_size={args.batch_size}",
                f"--num_workers={args.num_workers}",
                f"--gpu={args.gpu}",
                f"--seed={seed}",
                f"--save_root={save_root}",
            ]
            print("\n" + "=" * 100)
            print(f"running {run_name}")
            print(" ".join(cmd))
            print("=" * 100)

            return_code, log_text = run_once(cmd, run_log)
            metrics = extract_metrics(log_text)

            row = {
                "candidate_id": candidate_id,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "repeat": rep,
                "seed": seed,
                "run_log": str(run_log),
                "return_code": return_code,
            }

            if return_code == 0 and metrics is not None:
                row["status"] = "ok"
                row["f1"] = metrics["f1"]
                row["acc"] = metrics["acc"]
                row["val"] = metrics["val"]
                row["emoval"] = emoval_score(metrics["f1"], metrics["val"])
            else:
                row["status"] = "failed"
                row["f1"] = None
                row["acc"] = None
                row["val"] = None
                row["emoval"] = None
            raw_rows.append(row)

    summary = aggregate_results(raw_rows)

    raw_json = output_dir / f"raw_runs_{timestamp}.json"
    raw_csv = output_dir / f"raw_runs_{timestamp}.csv"
    summary_json = output_dir / f"summary_{timestamp}.json"
    summary_csv = output_dir / f"summary_{timestamp}.csv"

    raw_json.write_text(json.dumps(raw_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with raw_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(raw_rows[0].keys()) if raw_rows else [])
        if raw_rows:
            writer.writeheader()
            writer.writerows(raw_rows)

    with summary_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)

    print("\n" + "#" * 100)
    print("summary ranking (by mean_emoval)")
    for rank, item in enumerate(summary, start=1):
        print(
            f"[{rank}] {item['candidate_id']} "
            f"mean_emoval={item['mean_emoval']} std_emoval={item['std_emoval']} "
            f"mean_f1={item['mean_f1']} mean_val={item['mean_val']} "
            f"runs={item['runs_ok']}/{item['runs_total']}"
        )
    print("#" * 100)
    print(f"raw runs: {raw_json}")
    print(f"summary: {summary_json}")


if __name__ == "__main__":
    main()
