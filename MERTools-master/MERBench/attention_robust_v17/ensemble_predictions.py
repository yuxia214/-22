#!/usr/bin/env python3
"""
ensemble_predictions.py — 跨 seed / 跨 config 集成预测
用法:
    python ensemble_predictions.py \
        --result_dirs dir1 dir2 dir3 \
        --test_set test1 \
        [--weights 0.4 0.3 0.3] \
        [--output_dir ./ensemble_output]
"""
import argparse
import glob
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


def find_npz_files(result_dir, test_set="test1"):
    """在 result_dir 下递归查找匹配 test_set 的 .npz 文件"""
    pattern = os.path.join(result_dir, "**", f"{test_set}_*.npz")
    files = glob.glob(pattern, recursive=True)
    # 过滤掉没有 emo_probs 的旧格式文件
    valid = []
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            if "emo_probs" in data:
                valid.append(f)
            data.close()
        except Exception:
            pass
    return sorted(valid)


def load_predictions(npz_path):
    """加载单个 npz 的预测结果"""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "emo_probs": data["emo_probs"],
        "emo_labels": data["emo_labels"],
        "val_preds": data["val_preds"],
        "val_labels": data["val_labels"],
    }


def evaluate(emo_probs, emo_labels, val_preds, val_labels):
    """计算 WAF, ACC, MSE, Combined"""
    emo_preds = np.argmax(emo_probs, axis=1)
    acc = accuracy_score(emo_labels, emo_preds)
    f1 = f1_score(emo_labels, emo_preds, average="weighted")
    mse = mean_squared_error(val_labels, val_preds)
    combined = f1 - 0.25 * mse
    return {"f1": f1, "acc": acc, "mse": mse, "combined": combined}


def ensemble_average(all_preds, weights=None):
    """对多组预测做加权平均"""
    n = len(all_preds)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        s = sum(weights)
        weights = [w / s for w in weights]

    ref = all_preds[0]
    emo_labels = ref["emo_labels"]
    val_labels = ref["val_labels"]

    # 概率平均 (softmax 后平均)
    emo_probs = sum(w * p["emo_probs"] for w, p in zip(weights, all_preds))
    # 回归值平均
    val_preds = sum(w * p["val_preds"] for w, p in zip(weights, all_preds))

    return emo_probs, emo_labels, val_preds, val_labels


def main():
    parser = argparse.ArgumentParser(description="Ensemble predictions")
    parser.add_argument("--result_dirs", nargs="+", required=True,
                        help="结果目录列表 (每个目录下有 seed*/test*.npz)")
    parser.add_argument("--test_set", default="test1", help="test1 or test2")
    parser.add_argument("--weights", nargs="*", type=float, default=None,
                        help="各目录权重 (默认等权)")
    parser.add_argument("--output_dir", default=None,
                        help="保存集成结果的目录")
    args = parser.parse_args()

    print(f"=== Ensemble: {args.test_set} ===")
    print(f"Dirs: {args.result_dirs}")

    # 收集所有预测
    all_preds = []
    all_sources = []
    for d in args.result_dirs:
        files = find_npz_files(d, args.test_set)
        if not files:
            print(f"  [WARN] No valid npz in {d}, skipping")
            continue
        for f in files:
            pred = load_predictions(f)
            all_preds.append(pred)
            all_sources.append(f)
            metrics = evaluate(**pred)
            print(f"  {os.path.basename(f)}: "
                  f"F1={metrics['f1']:.4f} ACC={metrics['acc']:.4f} "
                  f"MSE={metrics['mse']:.4f} Comb={metrics['combined']:.6f}")

    if not all_preds:
        print("ERROR: No predictions found!")
        sys.exit(1)

    print(f"\nTotal models: {len(all_preds)}")

    # 等权集成
    emo_probs, emo_labels, val_preds, val_labels = ensemble_average(
        all_preds, args.weights
    )
    metrics = evaluate(emo_probs, emo_labels, val_preds, val_labels)
    print(f"\n>>> ENSEMBLE RESULT:")
    print(f"    F1      = {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    print(f"    ACC     = {metrics['acc']:.4f} ({metrics['acc']*100:.2f}%)")
    print(f"    MSE     = {metrics['mse']:.4f}")
    print(f"    Combined= {metrics['combined']:.6f}")

    # 保存集成结果
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(
            args.output_dir,
            f"ensemble_{args.test_set}_f1{metrics['f1']:.4f}_"
            f"mse{metrics['mse']:.4f}_comb{metrics['combined']:.6f}.npz"
        )
        np.savez_compressed(
            out_path,
            emo_probs=emo_probs, emo_labels=emo_labels,
            val_preds=val_preds, val_labels=val_labels,
            sources=np.array(all_sources, dtype=object),
        )
        print(f"    Saved to {out_path}")


if __name__ == "__main__":
    main()
