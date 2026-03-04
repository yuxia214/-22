#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MERBENCH_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if MERBENCH_ROOT not in sys.path:
    sys.path.insert(0, MERBENCH_ROOT)

from toolkit.dataloader import get_dataloaders
from toolkit.globals import idx2emo_mer
from toolkit.models import get_models


ALL_MODALITY_KEYS = ("audios", "texts", "videos")
MODE_KEEP_MAP = {
    "full": ("audios", "texts", "videos"),
    "av": ("audios", "videos"),
    "at": ("audios", "texts"),
    "vt": ("videos", "texts"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def build_masked_batch(batch: Dict[str, torch.Tensor], keep_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    keep_set = set(keep_keys)
    out = {}
    for key in ALL_MODALITY_KEYS:
        value = batch[key]
        out[key] = value if key in keep_set else torch.zeros_like(value)
    return out


def get_checkpoint_path(checkpoint_dir: str, fold_idx: int, seed: int) -> str:
    return os.path.join(checkpoint_dir, f"attention_robust_v7_seed{seed}_fold{fold_idx}.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AV/AT/VT accuracies using v7 checkpoints.")
    parser.add_argument("--output_dir", type=str, default="./attention_robust_v7/outputs/modality_combo_eval")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./attention_robust_v7/outputs/human_compare/models",
        help="directory containing attention_robust_v7_seed{seed}_fold{k}.pt",
    )
    parser.add_argument("--test_splits", type=str, default="all", help="test1,test2,test3 or all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)

    parser.add_argument("--dataset", type=str, default="MER2023")
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--test_dataset", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--audio_feature", type=str, default="chinese-hubert-large-UTT")
    parser.add_argument("--text_feature", type=str, default="Baichuan-13B-Base-UTT")
    parser.add_argument("--video_feature", type=str, default="clip-vit-large-patch14-UTT")
    parser.add_argument("--feat_type", type=str, default="utt")
    parser.add_argument("--feat_scale", type=int, default=1)
    parser.add_argument("--e2e_name", type=str, default=None)
    parser.add_argument("--e2e_dim", type=int, default=None)
    parser.add_argument("--model", type=str, default="attention_robust_v7")
    parser.add_argument("--n_classes", type=int, default=None)
    parser.add_argument("--hyper_path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--l2", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--print_iters", type=float, default=1e8)

    parser.add_argument("--emo_loss_weight", type=float, default=1.0)
    parser.add_argument("--val_loss_weight", type=float, default=1.3)
    parser.add_argument("--reg_loss_type", type=str, default="smoothl1", choices=["mse", "smoothl1"])
    parser.add_argument("--huber_beta", type=float, default=0.8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--modality_dropout", type=float, default=0.18)
    parser.add_argument("--use_modality_dropout", action="store_true", default=True)
    parser.add_argument("--modality_dropout_warmup", type=int, default=15)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--lr_factor", type=float, default=0.5)

    parser.add_argument("--use_vae", action="store_true", default=True)
    parser.add_argument("--kl_weight", type=float, default=0.01)
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument("--cross_kl_weight", type=float, default=0.01)
    parser.add_argument("--use_proxy_attention", action="store_true", default=True)
    parser.add_argument("--fusion_temperature", type=float, default=1.0)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--use_dynamic_kl", action="store_true", default=True)
    parser.add_argument("--kl_warmup_epochs", type=int, default=20)
    parser.add_argument("--use_valence_prior", action="store_true", default=True)
    parser.add_argument("--valence_consistency_weight", type=float, default=0.12)
    parser.add_argument("--valence_center_reg_weight", type=float, default=0.005)
    parser.add_argument("--feature_noise_std", type=float, default=0.03)
    parser.add_argument("--feature_noise_prob", type=float, default=0.35)
    parser.add_argument("--feature_noise_warmup", type=int, default=5)
    return parser.parse_args()


def parse_splits(raw: str) -> List[str]:
    text = raw.strip().lower()
    if text == "all":
        return ["test1", "test2", "test3"]
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    valid = {"test1", "test2", "test3"}
    if not items:
        raise ValueError("--test_splits is empty")
    for item in items:
        if item not in valid:
            raise ValueError(f"invalid split: {item}, valid: test1,test2,test3,all")
    return items


def infer_one_loader(model, dataloader, device: torch.device) -> Dict[str, np.ndarray]:
    names: List[str] = []
    true_emo: List[np.ndarray] = []

    emo_logits = {mode: [] for mode in MODE_KEEP_MAP}

    model.eval()
    with torch.no_grad():
        for batch, emos, vals, bnames in dataloader:
            names.extend(bnames)
            for key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)
            true_emo.append(emos.numpy())

            for mode, keep in MODE_KEEP_MAP.items():
                masked_batch = build_masked_batch(batch, keep)
                _, emo_out, _, _ = model(masked_batch)
                emo_logits[mode].append(emo_out.detach().cpu().numpy())

    result: Dict[str, np.ndarray] = {
        "names": np.array(names, dtype=object),
        "true_emo": np.concatenate(true_emo, axis=0).astype(np.int64),
    }
    for mode in MODE_KEEP_MAP:
        result[f"{mode}_emo_logits"] = np.concatenate(emo_logits[mode], axis=0).astype(np.float32)
    return result


def ensemble_fold_outputs(fold_outputs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not fold_outputs:
        raise RuntimeError("empty fold outputs")
    base_names = fold_outputs[0]["names"]
    base_true_emo = fold_outputs[0]["true_emo"]

    for idx, each in enumerate(fold_outputs[1:], start=2):
        if not np.array_equal(base_names, each["names"]):
            raise RuntimeError(f"name order mismatch across folds, fold {idx}")
        if not np.array_equal(base_true_emo, each["true_emo"]):
            raise RuntimeError(f"emotion labels mismatch across folds, fold {idx}")

    out: Dict[str, np.ndarray] = {
        "names": base_names,
        "true_emo": base_true_emo,
    }
    for mode in MODE_KEEP_MAP:
        emo_stack = np.stack([x[f"{mode}_emo_logits"] for x in fold_outputs], axis=0)
        out[f"{mode}_emo_logits"] = np.mean(emo_stack, axis=0)
    return out


def build_split_detail_df(split_name: str, ensemble_out: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = ensemble_out["names"]
    true_emo = ensemble_out["true_emo"]
    preds = {}
    for mode in MODE_KEEP_MAP:
        preds[mode] = np.argmax(ensemble_out[f"{mode}_emo_logits"], axis=1).astype(np.int64)

    df = pd.DataFrame(
        {
            "split": split_name,
            "clip_name": names,
            "true_label": [idx2emo_mer[int(x)] for x in true_emo],
            "pred_full": [idx2emo_mer[int(x)] for x in preds["full"]],
            "pred_av": [idx2emo_mer[int(x)] for x in preds["av"]],
            "pred_at": [idx2emo_mer[int(x)] for x in preds["at"]],
            "pred_vt": [idx2emo_mer[int(x)] for x in preds["vt"]],
            "correct_full": (preds["full"] == true_emo).astype(np.int64),
            "correct_av": (preds["av"] == true_emo).astype(np.int64),
            "correct_at": (preds["at"] == true_emo).astype(np.int64),
            "correct_vt": (preds["vt"] == true_emo).astype(np.int64),
        }
    )
    return df


def split_accuracy_row(split_name: str, detail_df: pd.DataFrame) -> Dict[str, float]:
    row = {
        "split": split_name,
        "samples": int(len(detail_df)),
        "acc_full": float(detail_df["correct_full"].mean()) if len(detail_df) > 0 else 0.0,
        "acc_av": float(detail_df["correct_av"].mean()) if len(detail_df) > 0 else 0.0,
        "acc_at": float(detail_df["correct_at"].mean()) if len(detail_df) > 0 else 0.0,
        "acc_vt": float(detail_df["correct_vt"].mean()) if len(detail_df) > 0 else 0.0,
    }
    row["delta_av_vs_full"] = row["acc_av"] - row["acc_full"]
    row["delta_at_vs_full"] = row["acc_at"] - row["acc_full"]
    row["delta_vt_vs_full"] = row["acc_vt"] - row["acc_full"]
    return row


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_path(args.output_dir, MERBENCH_ROOT)
    args.checkpoint_dir = resolve_path(args.checkpoint_dir, MERBENCH_ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    selected_splits = parse_splits(args.test_splits)

    if args.feat_type == "utt":
        args.feat_scale = 1
    elif args.feat_type == "frm_align":
        args.feat_scale = 6
    elif args.feat_type == "frm_unalign":
        args.feat_scale = 12

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("====== Reading Data ======")
    dataloader_class = get_dataloaders(args)
    train_loaders, _, test_loaders = dataloader_class.get_loaders()
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()

    split_loader_map = {
        "test1": test_loaders[0],
        "test2": test_loaders[1],
        "test3": test_loaders[2],
    }
    fold_outputs_by_split: Dict[str, List[Dict[str, np.ndarray]]] = {k: [] for k in selected_splits}

    print("====== Inference with checkpoints ======")
    run_start = time.time()
    for fold_idx in range(1, args.num_folds + 1):
        ckpt_path = get_checkpoint_path(args.checkpoint_dir, fold_idx, args.seed)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        model = get_models(args).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)

        for split_name in selected_splits:
            loader = split_loader_map[split_name]
            fold_out = infer_one_loader(model, loader, device)
            fold_outputs_by_split[split_name].append(fold_out)
            print(f"fold {fold_idx} done: {split_name}, samples={len(fold_out['names'])}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    split_rows = []
    detail_paths = []

    print("====== Ensembling + accuracy summary ======")
    for split_name in selected_splits:
        ens_out = ensemble_fold_outputs(fold_outputs_by_split[split_name])
        detail_df = build_split_detail_df(split_name, ens_out)
        split_row = split_accuracy_row(split_name, detail_df)
        split_rows.append(split_row)

        detail_path = os.path.join(args.output_dir, f"{split_name}_combo_detail_{run_tag}.csv")
        detail_df.to_csv(detail_path, index=False)
        detail_paths.append(detail_path)
        print(
            f"{split_name}: full={split_row['acc_full']:.4f} "
            f"av={split_row['acc_av']:.4f} at={split_row['acc_at']:.4f} vt={split_row['acc_vt']:.4f}"
        )

    split_df = pd.DataFrame(split_rows)
    total_samples = int(split_df["samples"].sum())
    if total_samples > 0:
        overall_row = {
            "split": "overall_weighted",
            "samples": total_samples,
            "acc_full": float(np.average(split_df["acc_full"], weights=split_df["samples"])),
            "acc_av": float(np.average(split_df["acc_av"], weights=split_df["samples"])),
            "acc_at": float(np.average(split_df["acc_at"], weights=split_df["samples"])),
            "acc_vt": float(np.average(split_df["acc_vt"], weights=split_df["samples"])),
        }
        overall_row["delta_av_vs_full"] = overall_row["acc_av"] - overall_row["acc_full"]
        overall_row["delta_at_vs_full"] = overall_row["acc_at"] - overall_row["acc_full"]
        overall_row["delta_vt_vs_full"] = overall_row["acc_vt"] - overall_row["acc_full"]
        split_df = pd.concat([split_df, pd.DataFrame([overall_row])], ignore_index=True)

    split_df["duration_seconds"] = time.time() - run_start
    summary_path = os.path.join(args.output_dir, f"combo_accuracy_summary_{run_tag}.csv")
    split_df.to_csv(summary_path, index=False)

    print("====== Done ======")
    print(f"summary: {summary_path}")
    for path in detail_paths:
        print(f"detail: {path}")


if __name__ == "__main__":
    main()
