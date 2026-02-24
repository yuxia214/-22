#!/usr/bin/env python3
"""
ensemble_predictions_v19.py

Key ideas:
1) Separate weights for classification probs and regression preds.
2) Sparse forward selection to keep only complementary models.
3) Random refinement on top of sparse solution.
"""
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


EPS = 1e-12


@dataclass
class PredPack:
    emo_probs: np.ndarray
    emo_labels: np.ndarray
    val_preds: np.ndarray
    val_labels: np.ndarray


@dataclass
class ModelPack:
    result_dir: str
    files: Dict[str, str]
    sets: Dict[str, PredPack]


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w[w < 0.0] = 0.0
    s = float(np.sum(w))
    if s <= EPS:
        return np.ones_like(w) / max(1, len(w))
    return w / s


def find_latest_npz_for_set(result_dir: str, test_set: str) -> Optional[str]:
    pattern = os.path.join(result_dir, "**", f"{test_set}_*.npz")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _pick_key(data, candidates: List[str]):
    for key in candidates:
        if key in data:
            return data[key]
    return None


def _to_probs_if_hard_labels(arr: np.ndarray, emo_labels: np.ndarray) -> Optional[np.ndarray]:
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


def load_prediction_file(npz_path: str) -> Tuple[Optional[PredPack], Optional[str]]:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return None, f"np.load failed: {e}"

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

    return (
        PredPack(
            emo_probs=emo_probs.astype(np.float32),
            emo_labels=emo_labels,
            val_preds=val_preds.astype(np.float32),
            val_labels=val_labels,
        ),
        None,
    )


def evaluate_pack(emo_probs: np.ndarray, emo_labels: np.ndarray, val_preds: np.ndarray, val_labels: np.ndarray) -> Dict[str, float]:
    emo_preds = np.argmax(emo_probs, axis=1)
    acc = accuracy_score(emo_labels, emo_preds)
    f1 = f1_score(emo_labels, emo_preds, average="weighted")
    mse = mean_squared_error(val_labels, val_preds)
    combined = f1 - 0.25 * mse
    return {
        "f1": float(f1),
        "acc": float(acc),
        "mse": float(mse),
        "combined": float(combined),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    return (
        f"F1={metrics['f1']:.4f} ACC={metrics['acc']:.4f} "
        f"MSE={metrics['mse']:.4f} Combined={metrics['combined']:.6f}"
    )


def collect_models(result_dirs: List[str], all_sets: List[str], tune_set: str) -> List[ModelPack]:
    models: List[ModelPack] = []

    for d in result_dirs:
        files: Dict[str, str] = {}
        sets: Dict[str, PredPack] = {}

        for set_name in all_sets:
            npz_path = find_latest_npz_for_set(d, set_name)
            if npz_path is None:
                continue
            pack, err = load_prediction_file(npz_path)
            if pack is None:
                print(f"  [WARN] skip {npz_path}: {err}")
                continue
            files[set_name] = npz_path
            sets[set_name] = pack

        if tune_set not in sets:
            print(f"  [WARN] skip model dir (no valid {tune_set}): {d}")
            continue

        models.append(ModelPack(result_dir=d, files=files, sets=sets))

    return models


def _subset_indices_with_set(models: List[ModelPack], set_name: str) -> List[int]:
    return [i for i, m in enumerate(models) if set_name in m.sets]


def _validate_labels_consistency(models: List[ModelPack], idxs: List[int], set_name: str) -> None:
    ref = models[idxs[0]].sets[set_name]
    for i in idxs[1:]:
        cur = models[i].sets[set_name]
        if not np.array_equal(ref.emo_labels, cur.emo_labels):
            raise ValueError(f"emo_labels mismatch on {set_name}: model idx {idxs[0]} vs {i}")
        if not np.array_equal(ref.val_labels, cur.val_labels):
            raise ValueError(f"val_labels mismatch on {set_name}: model idx {idxs[0]} vs {i}")


def combine_on_set(
    models: List[ModelPack],
    set_name: str,
    w_cls: np.ndarray,
    w_reg: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray]:
    idxs = _subset_indices_with_set(models, set_name)
    if not idxs:
        raise ValueError(f"no models contain set={set_name}")

    _validate_labels_consistency(models, idxs, set_name)

    wc = normalize_weights(np.asarray([w_cls[i] for i in idxs], dtype=np.float64))
    wr = normalize_weights(np.asarray([w_reg[i] for i in idxs], dtype=np.float64))

    ref = models[idxs[0]].sets[set_name]
    emo_probs = np.zeros_like(ref.emo_probs, dtype=np.float64)
    val_preds = np.zeros_like(ref.val_preds, dtype=np.float64)

    for wi, i in zip(wc, idxs):
        emo_probs += wi * models[i].sets[set_name].emo_probs
    for wi, i in zip(wr, idxs):
        val_preds += wi * models[i].sets[set_name].val_preds

    emo_probs = emo_probs.astype(np.float32)
    val_preds = val_preds.astype(np.float32)
    metrics = evaluate_pack(emo_probs, ref.emo_labels, val_preds, ref.val_labels)
    return metrics, emo_probs, ref.emo_labels, val_preds, ref.val_labels, idxs, wc, wr


def tune_score(models: List[ModelPack], tune_set: str, w_cls: np.ndarray, w_reg: np.ndarray) -> Tuple[float, Dict[str, float]]:
    metrics, _, _, _, _, _, _, _ = combine_on_set(models, tune_set, w_cls, w_reg)
    return float(metrics["combined"]), metrics


def _safe_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a).reshape(-1).astype(np.float64)
    bb = np.asarray(b).reshape(-1).astype(np.float64)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    if np.std(aa) <= EPS or np.std(bb) <= EPS:
        return 1.0 if np.allclose(aa, bb) else 0.0
    corr = np.corrcoef(aa, bb)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(abs(corr))


def candidate_penalty(
    models: List[ModelPack],
    tune_set: str,
    selected: List[int],
    candidate: int,
    corr_penalty: float,
    corr_threshold: float,
) -> float:
    if corr_penalty <= 0 or not selected:
        return 0.0

    cand = models[candidate].sets[tune_set]
    corrs = []
    for i in selected:
        ref = models[i].sets[tune_set]
        corrs.append(_safe_abs_corr(ref.val_preds, cand.val_preds))
    mean_corr = float(np.mean(corrs)) if corrs else 0.0
    overflow = max(0.0, mean_corr - corr_threshold)
    return float(corr_penalty * overflow)


def forward_sparse_dual_search(
    models: List[ModelPack],
    tune_set: str,
    max_models: int,
    alpha_grid: List[float],
    min_improve: float,
    corr_penalty: float,
    corr_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, float]]:
    n = len(models)
    if n == 0:
        raise ValueError("empty models")

    # Best single model as start.
    single = []
    for i in range(n):
        wc = np.zeros(n, dtype=np.float64)
        wr = np.zeros(n, dtype=np.float64)
        wc[i] = 1.0
        wr[i] = 1.0
        score, m = tune_score(models, tune_set, wc, wr)
        single.append((score, i, m))
    single.sort(reverse=True, key=lambda x: x[0])

    best_idx = single[0][1]
    wc_best = np.zeros(n, dtype=np.float64)
    wr_best = np.zeros(n, dtype=np.float64)
    wc_best[best_idx] = 1.0
    wr_best[best_idx] = 1.0
    score_best, metrics_best = tune_score(models, tune_set, wc_best, wr_best)
    selected = [best_idx]

    while len(selected) < min(max_models, n):
        pick = None
        pick_adj = score_best

        for cand in range(n):
            if cand in selected:
                continue

            penalty = candidate_penalty(
                models=models,
                tune_set=tune_set,
                selected=selected,
                candidate=cand,
                corr_penalty=corr_penalty,
                corr_threshold=corr_threshold,
            )
            onehot = np.zeros(n, dtype=np.float64)
            onehot[cand] = 1.0

            local_best = None
            local_best_adj = -1e18
            for a_cls in alpha_grid:
                for a_reg in alpha_grid:
                    wc_trial = normalize_weights((1.0 - a_cls) * wc_best + a_cls * onehot)
                    wr_trial = normalize_weights((1.0 - a_reg) * wr_best + a_reg * onehot)
                    score_trial, metrics_trial = tune_score(models, tune_set, wc_trial, wr_trial)
                    adj = score_trial - penalty
                    if adj > local_best_adj + 1e-12:
                        local_best_adj = adj
                        local_best = (score_trial, metrics_trial, wc_trial, wr_trial)

            if local_best is None:
                continue

            raw_score, raw_metrics, raw_wc, raw_wr = local_best
            if raw_score <= score_best + min_improve:
                continue
            if local_best_adj > pick_adj + 1e-12:
                pick_adj = local_best_adj
                pick = (cand, raw_score, raw_metrics, raw_wc, raw_wr)

        if pick is None:
            break

        cand, score_best, metrics_best, wc_best, wr_best = pick
        selected.append(cand)

    return wc_best, wr_best, selected, metrics_best


def random_refine_dual(
    models: List[ModelPack],
    tune_set: str,
    w_cls_init: np.ndarray,
    w_reg_init: np.ndarray,
    search_trials: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(w_cls_init)

    wc_best = normalize_weights(w_cls_init)
    wr_best = normalize_weights(w_reg_init)
    score_best, metrics_best = tune_score(models, tune_set, wc_best, wr_best)

    active = np.where((wc_best > 1e-10) | (wr_best > 1e-10))[0]
    if active.size == 0:
        active = np.arange(n)
    n_active = int(active.size)

    for t in range(max(0, int(search_trials))):
        if t % 11 == 0:
            idx = np.arange(n)
        elif t % 5 == 0:
            idx = active
        else:
            # Guard corner case: sparse solution may keep only one active model.
            if n_active <= 1:
                idx = active
            else:
                k = int(rng.integers(2, n_active + 1))
                idx = rng.choice(active, size=k, replace=False)

        sample_cls = rng.dirichlet(np.ones(len(idx), dtype=np.float64))
        sample_reg = rng.dirichlet(np.ones(len(idx), dtype=np.float64))

        wc_trial = np.zeros(n, dtype=np.float64)
        wr_trial = np.zeros(n, dtype=np.float64)
        wc_trial[idx] = sample_cls
        wr_trial[idx] = sample_reg

        mix_cls = float(rng.uniform(0.30, 1.00))
        mix_reg = float(rng.uniform(0.30, 1.00))
        wc_trial = normalize_weights(mix_cls * wc_trial + (1.0 - mix_cls) * wc_best)
        wr_trial = normalize_weights(mix_reg * wr_trial + (1.0 - mix_reg) * wr_best)

        score_trial, metrics_trial = tune_score(models, tune_set, wc_trial, wr_trial)
        if score_trial > score_best + 1e-10:
            score_best = score_trial
            metrics_best = metrics_trial
            wc_best = wc_trial
            wr_best = wr_trial

    return wc_best, wr_best, metrics_best


def main() -> None:
    parser = argparse.ArgumentParser(description="v19 ensemble: dual weights + sparse search")
    parser.add_argument("--result_dirs", nargs="+", required=True, help="model result directories")
    parser.add_argument("--tune_set", default="test1", help="set used to optimize weights")
    parser.add_argument("--eval_sets", nargs="+", default=["test1", "test2", "test3"], help="sets to report/save")
    parser.add_argument("--max_models", type=int, default=8, help="max sparse selected models")
    parser.add_argument("--alpha_grid", type=str, default="0.05,0.10,0.15,0.20,0.30,0.40,0.50", help="grid for forward add")
    parser.add_argument("--min_improve", type=float, default=5e-5, help="minimum raw combined improvement for adding a model")
    parser.add_argument("--corr_penalty", type=float, default=0.0, help="penalty coefficient for overly correlated reg predictions")
    parser.add_argument("--corr_threshold", type=float, default=0.985, help="correlation threshold before penalty")
    parser.add_argument("--search_trials", type=int, default=5000, help="random refinement trials")
    parser.add_argument("--seed", type=int, default=42, help="random seed for refinement")
    parser.add_argument("--output_dir", default=None, help="directory to save ensemble npz/json")
    parser.add_argument("--save_prefix", default="v19", help="output file prefix")
    args = parser.parse_args()

    all_sets = list(dict.fromkeys([args.tune_set] + list(args.eval_sets)))

    print("=== v19 Ensemble ===")
    print(f"tune_set={args.tune_set}")
    print(f"eval_sets={args.eval_sets}")
    print(f"result_dirs={args.result_dirs}")

    models = collect_models(args.result_dirs, all_sets=all_sets, tune_set=args.tune_set)
    if not models:
        print("ERROR: no valid model packs found")
        sys.exit(1)

    print(f"\nValid model packs: {len(models)}")
    for i, m in enumerate(models):
        tune_file = m.files.get(args.tune_set, "")
        single = m.sets[args.tune_set]
        metrics = evaluate_pack(single.emo_probs, single.emo_labels, single.val_preds, single.val_labels)
        print(f"  [{i:02d}] {m.result_dir} -> {format_metrics(metrics)}")
        print(f"       file: {tune_file}")

    alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()]
    if not alpha_grid:
        alpha_grid = [0.1, 0.2, 0.3, 0.4]

    w_cls_sparse, w_reg_sparse, selected, sparse_metrics = forward_sparse_dual_search(
        models=models,
        tune_set=args.tune_set,
        max_models=max(1, args.max_models),
        alpha_grid=alpha_grid,
        min_improve=max(0.0, args.min_improve),
        corr_penalty=max(0.0, args.corr_penalty),
        corr_threshold=float(args.corr_threshold),
    )
    print("\nSparse search result:")
    print(f"  selected={selected}")
    print(f"  tune_metrics={format_metrics(sparse_metrics)}")

    w_cls_final, w_reg_final, refine_metrics = random_refine_dual(
        models=models,
        tune_set=args.tune_set,
        w_cls_init=w_cls_sparse,
        w_reg_init=w_reg_sparse,
        search_trials=max(0, int(args.search_trials)),
        seed=int(args.seed),
    )
    print("\nRefined result:")
    print(f"  tune_metrics={format_metrics(refine_metrics)}")

    # Show top weights for readability.
    top_cls = sorted([(float(w), i) for i, w in enumerate(w_cls_final) if w > 1e-6], reverse=True)
    top_reg = sorted([(float(w), i) for i, w in enumerate(w_reg_final) if w > 1e-6], reverse=True)
    print("\nTop cls weights:")
    for w, i in top_cls[:20]:
        print(f"  w_cls={w:.6f} idx={i} dir={models[i].result_dir}")
    print("Top reg weights:")
    for w, i in top_reg[:20]:
        print(f"  w_reg={w:.6f} idx={i} dir={models[i].result_dir}")

    report = {
        "tune_set": args.tune_set,
        "eval_sets": args.eval_sets,
        "tune_metrics": refine_metrics,
        "models": [m.result_dir for m in models],
        "weights_cls": [float(x) for x in w_cls_final.tolist()],
        "weights_reg": [float(x) for x in w_reg_final.tolist()],
        "selected_sparse": [int(x) for x in selected],
    }

    for set_name in args.eval_sets:
        try:
            metrics, emo_probs, emo_labels, val_preds, val_labels, idxs, wc_sub, wr_sub = combine_on_set(
                models,
                set_name,
                w_cls_final,
                w_reg_final,
            )
        except Exception as e:
            print(f"\n[WARN] skip eval set {set_name}: {e}")
            continue

        print(f"\n[{set_name}] {format_metrics(metrics)}")
        report[f"metrics_{set_name}"] = metrics
        report[f"active_indices_{set_name}"] = [int(x) for x in idxs]
        report[f"active_w_cls_{set_name}"] = [float(x) for x in wc_sub.tolist()]
        report[f"active_w_reg_{set_name}"] = [float(x) for x in wr_sub.tolist()]

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            npz_path = os.path.join(
                args.output_dir,
                f"{args.save_prefix}_{set_name}_f1{metrics['f1']:.4f}_mse{metrics['mse']:.4f}_comb{metrics['combined']:.6f}.npz",
            )
            np.savez_compressed(
                npz_path,
                emo_probs=emo_probs,
                emo_labels=emo_labels,
                val_preds=val_preds,
                val_labels=val_labels,
                weights_cls=w_cls_final.astype(np.float32),
                weights_reg=w_reg_final.astype(np.float32),
                model_dirs=np.array([m.result_dir for m in models], dtype=object),
            )
            print(f"  saved npz: {npz_path}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, f"{args.save_prefix}_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nSaved summary: {json_path}")


if __name__ == "__main__":
    main()
