#!/usr/bin/env python3
"""
ensemble_predictions.py - 多模型预测集成（支持自动加权搜索）

典型用法:
    python ensemble_predictions.py \
        --result_dirs dir1 dir2 dir3 \
        --test_set test1 \
        --auto_search \
        --search_trials 4000 \
        --output_dir ./ensemble_output
"""
import argparse
import glob
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


EPS = 1e-12


def find_npz_files(result_dir, test_set="test1"):
    """递归查找 result_dir 下对应 test_set 的结果文件。"""
    pattern = os.path.join(result_dir, "**", f"{test_set}_*.npz")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return []

    # 同一目录下如果有多个重跑文件，只保留最新文件，避免重复计权。
    latest_by_parent = {}
    for path in files:
        parent = os.path.dirname(path)
        old = latest_by_parent.get(parent)
        if old is None or os.path.getmtime(path) >= os.path.getmtime(old):
            latest_by_parent[parent] = path
    return sorted(latest_by_parent.values())


def _pick_key(data, candidates):
    for key in candidates:
        if key in data:
            return data[key]
    return None


def _to_probs_if_hard_labels(arr, emo_labels):
    arr = np.asarray(arr)
    if arr.ndim != 1:
        return arr

    hard = arr.astype(np.int64)
    if hard.size == 0:
        return None
    max_label = int(max(np.max(hard), np.max(np.asarray(emo_labels).astype(np.int64))))
    n_classes = max_label + 1
    probs = np.zeros((hard.shape[0], n_classes), dtype=np.float32)
    probs[np.arange(hard.shape[0]), hard] = 1.0
    return probs


def load_predictions(npz_path):
    """加载单个 npz 预测文件，兼容多种键名。"""
    data = np.load(npz_path, allow_pickle=True)

    emo_labels = _pick_key(data, ["emo_labels", "emolabels"])
    emo_probs = _pick_key(data, ["emo_probs", "emoprobs", "emo_preds", "emopreds"])
    val_preds = _pick_key(data, ["val_preds", "valpreds"])
    val_labels = _pick_key(data, ["val_labels", "vallabels"])

    data.close()

    missing = []
    if emo_labels is None:
        missing.append("emo_labels/emolabels")
    if emo_probs is None:
        missing.append("emo_probs/emoprobs")
    if val_preds is None:
        missing.append("val_preds/valpreds")
    if val_labels is None:
        missing.append("val_labels/vallabels")
    if missing:
        return None, f"missing keys: {', '.join(missing)}"

    emo_labels = np.asarray(emo_labels)
    val_preds = np.asarray(val_preds)
    val_labels = np.asarray(val_labels)
    emo_probs = _to_probs_if_hard_labels(emo_probs, emo_labels)
    if emo_probs is None:
        return None, "empty emo predictions"
    emo_probs = np.asarray(emo_probs)

    if emo_probs.ndim != 2:
        return None, f"emo_probs ndim={emo_probs.ndim}, expected 2"
    n = emo_probs.shape[0]
    if emo_labels.shape[0] != n or val_preds.shape[0] != n or val_labels.shape[0] != n:
        return None, (
            "sample size mismatch: "
            f"emo_probs={n}, emo_labels={emo_labels.shape[0]}, "
            f"val_preds={val_preds.shape[0]}, val_labels={val_labels.shape[0]}"
        )

    return {
        "emo_probs": emo_probs.astype(np.float32),
        "emo_labels": emo_labels,
        "val_preds": val_preds.astype(np.float32),
        "val_labels": val_labels,
    }, None


def evaluate(emo_probs, emo_labels, val_preds, val_labels):
    """计算 WAF/F1, ACC, MSE, Combined。"""
    emo_preds = np.argmax(emo_probs, axis=1)
    acc = accuracy_score(emo_labels, emo_preds)
    f1 = f1_score(emo_labels, emo_preds, average="weighted")
    mse = mean_squared_error(val_labels, val_preds)
    combined = f1 - 0.25 * mse
    return {"f1": float(f1), "acc": float(acc), "mse": float(mse), "combined": float(combined)}


def normalize_weights(weights):
    w = np.asarray(weights, dtype=np.float64)
    w[w < 0.0] = 0.0
    s = float(np.sum(w))
    if s <= EPS:
        w = np.ones_like(w) / max(1, len(w))
    else:
        w = w / s
    return w


def combine_predictions(models, weights):
    weights = normalize_weights(weights)
    ref = models[0]
    emo_labels = ref["emo_labels"]
    val_labels = ref["val_labels"]

    for m in models[1:]:
        if not np.array_equal(m["emo_labels"], emo_labels):
            raise ValueError("emo_labels mismatch across models; cannot ensemble")
        if not np.array_equal(m["val_labels"], val_labels):
            raise ValueError("val_labels mismatch across models; cannot ensemble")

    emo_probs = np.zeros_like(ref["emo_probs"], dtype=np.float64)
    val_preds = np.zeros_like(ref["val_preds"], dtype=np.float64)
    for wi, mi in zip(weights, models):
        emo_probs += wi * mi["emo_probs"]
        val_preds += wi * mi["val_preds"]

    return emo_probs.astype(np.float32), emo_labels, val_preds.astype(np.float32), val_labels


def ensemble_metrics(models, weights):
    emo_probs, emo_labels, val_preds, val_labels = combine_predictions(models, weights)
    metrics = evaluate(emo_probs, emo_labels, val_preds, val_labels)
    return metrics, emo_probs, emo_labels, val_preds, val_labels


def greedy_weight_search(models):
    n = len(models)
    alphas = np.linspace(0.05, 0.95, 19)

    single_metrics = []
    for i in range(n):
        w = np.zeros(n, dtype=np.float64)
        w[i] = 1.0
        m, _, _, _, _ = ensemble_metrics(models, w)
        single_metrics.append((m["combined"], i, m))

    single_metrics.sort(reverse=True, key=lambda x: x[0])
    best_idx = single_metrics[0][1]

    cur_w = np.zeros(n, dtype=np.float64)
    cur_w[best_idx] = 1.0
    cur_m, _, _, _, _ = ensemble_metrics(models, cur_w)
    selected = {best_idx}

    while len(selected) < n:
        best_trial_w = None
        best_trial_m = cur_m
        best_trial_idx = None

        for i in range(n):
            if i in selected:
                continue
            basis = np.zeros(n, dtype=np.float64)
            basis[i] = 1.0
            for alpha in alphas:
                trial_w = normalize_weights((1.0 - alpha) * cur_w + alpha * basis)
                trial_m, _, _, _, _ = ensemble_metrics(models, trial_w)
                if trial_m["combined"] > best_trial_m["combined"] + 1e-10:
                    best_trial_m = trial_m
                    best_trial_w = trial_w
                    best_trial_idx = i

        if best_trial_w is None:
            break

        cur_w = best_trial_w
        cur_m = best_trial_m
        selected.add(best_trial_idx)

    return cur_w, cur_m, selected


def random_weight_refine(models, init_weights, n_trials=3000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(models)

    best_w = normalize_weights(init_weights)
    best_m, _, _, _, _ = ensemble_metrics(models, best_w)

    for t in range(max(0, int(n_trials))):
        active = np.where(best_w > 1e-10)[0]
        if active.size == 0:
            active = np.arange(n)

        if t % 6 == 0:
            idx = np.arange(n)
        else:
            idx = active

        sampled = rng.dirichlet(np.ones(len(idx), dtype=np.float64))
        trial = np.zeros(n, dtype=np.float64)
        trial[idx] = sampled

        mix = rng.uniform(0.35, 1.0)
        trial = normalize_weights(mix * trial + (1.0 - mix) * best_w)

        trial_m, _, _, _, _ = ensemble_metrics(models, trial)
        if trial_m["combined"] > best_m["combined"] + 1e-10:
            best_m = trial_m
            best_w = trial

    return best_w, best_m


def format_metrics(m):
    return (
        f"F1={m['f1']:.4f} ACC={m['acc']:.4f} "
        f"MSE={m['mse']:.4f} Combined={m['combined']:.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Ensemble predictions with optional auto weight search")
    parser.add_argument("--result_dirs", nargs="+", required=True, help="结果目录列表")
    parser.add_argument("--test_set", default="test1", help="test1/test2/test3")
    parser.add_argument("--weights", nargs="*", type=float, default=None, help="手动权重")
    parser.add_argument("--auto_search", action="store_true", help="启用自动权重搜索")
    parser.add_argument("--search_trials", type=int, default=3000, help="自动搜索随机细化次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_dir", default=None, help="保存目录")
    args = parser.parse_args()

    print(f"=== Ensemble: {args.test_set} ===")
    print(f"Result dirs: {args.result_dirs}")

    models = []
    sources = []
    skipped = 0

    for d in args.result_dirs:
        files = find_npz_files(d, args.test_set)
        if not files:
            print(f"  [WARN] No {args.test_set}_*.npz in {d}")
            continue

        for f in files:
            pred, err = load_predictions(f)
            if pred is None:
                print(f"  [WARN] Skip {f}: {err}")
                skipped += 1
                continue

            metrics = evaluate(
                pred["emo_probs"], pred["emo_labels"], pred["val_preds"], pred["val_labels"]
            )
            print(f"  {os.path.basename(f)} -> {format_metrics(metrics)}")
            models.append(pred)
            sources.append(f)

    if not models:
        print("ERROR: No valid prediction files found.")
        sys.exit(1)

    n = len(models)
    print(f"\nValid models: {n}, skipped: {skipped}")

    if args.weights is not None and len(args.weights) > 0:
        if len(args.weights) != n:
            print(f"ERROR: weights length={len(args.weights)} but valid models={n}")
            sys.exit(2)
        final_w = normalize_weights(args.weights)
        print("Using manual weights.")
    elif args.auto_search:
        greedy_w, greedy_m, selected = greedy_weight_search(models)
        print(
            f"Greedy search: selected={len(selected)} models, "
            f"{format_metrics(greedy_m)}"
        )
        final_w, final_m = random_weight_refine(
            models,
            init_weights=greedy_w,
            n_trials=args.search_trials,
            seed=args.seed,
        )
        print(f"Refined search: {format_metrics(final_m)}")
    else:
        final_w = normalize_weights(np.ones(n, dtype=np.float64))
        print("Using equal weights.")

    metrics, emo_probs, emo_labels, val_preds, val_labels = ensemble_metrics(models, final_w)
    print("\n>>> ENSEMBLE RESULT")
    print(f"    {format_metrics(metrics)}")

    nonzero = [(w, i) for i, w in enumerate(final_w.tolist()) if w > 1e-6]
    nonzero.sort(reverse=True, key=lambda x: x[0])
    print("\nTop weights:")
    for w, i in nonzero[:20]:
        print(f"    w={w:.6f} | {sources[i]}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(
            args.output_dir,
            f"ensemble_{args.test_set}_f1{metrics['f1']:.4f}_"
            f"mse{metrics['mse']:.4f}_comb{metrics['combined']:.6f}.npz",
        )
        np.savez_compressed(
            out_path,
            emo_probs=emo_probs,
            emo_labels=emo_labels,
            val_preds=val_preds,
            val_labels=val_labels,
            weights=final_w.astype(np.float32),
            sources=np.array(sources, dtype=object),
        )
        print(f"Saved ensemble file: {out_path}")


if __name__ == "__main__":
    main()
