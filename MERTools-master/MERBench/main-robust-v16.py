"""
改进版训练脚本 - 针对模态缺失场景(test2)优化
主要改进:
1. 使用AttentionRobust模型 (带模态dropout)
2. 添加早停机制 (Early Stopping)
3. 添加学习率调度器 (ReduceLROnPlateau)
4. 增强正则化 (更高的dropout和L2)
5. 固定超参数 (避免随机选择导致不稳定)

支持的模型:
- attention_robust: 带模态dropout的attention模型
- attention_robust_v2: 基于P-RMF的概率化多模态融合模型 (VAE + 不确定性加权 + 代理模态)

使用方法 (V1):
python -u main-robust.py --model='attention_robust' --feat_type='utt' --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --gpu=0

使用方法 (V2 - 推荐):
python -u main-robust.py --model='attention_robust_v2' --feat_type='utt' --dataset='MER2023' \
    --audio_feature='chinese-hubert-large-UTT' \
    --text_feature='Baichuan-13B-Base-UTT' \
    --video_feature='clip-vit-large-patch14-UTT' \
    --hidden_dim=128 --dropout=0.35 \
    --use_vae --kl_weight=0.01 --recon_weight=0.1 --cross_kl_weight=0.01 \
    --use_proxy_attention --fusion_temperature=1.0 \
    --modality_dropout=0.15 --modality_dropout_warmup=20 \
    --lr=5e-4 --l2=5e-5 --epochs=100 --early_stopping_patience=30 \
    --gpu=0
"""

import os
import time
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import mean_squared_error

from toolkit.utils.loss import *
from toolkit.utils.metric import *
from toolkit.utils.functions import *
from toolkit.models import get_models
from toolkit.dataloader import get_dataloaders


class EarlyStopping:
    """早停机制，防止过拟合"""
    def __init__(self, patience=15, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class UncertaintyWeightedTaskLoss(torch.nn.Module):
    """Homoscedastic uncertainty weighting for cls/reg multi-task losses."""
    def __init__(self, init_log_var_cls=0.0, init_log_var_reg=0.0):
        super().__init__()
        self.log_var_cls = torch.nn.Parameter(torch.tensor(float(init_log_var_cls)))
        self.log_var_reg = torch.nn.Parameter(torch.tensor(float(init_log_var_reg)))

    def forward(self, interloss, cls_term=None, reg_term=None):
        loss = interloss
        if cls_term is not None:
            precision_cls = torch.exp(-self.log_var_cls)
            loss = loss + precision_cls * cls_term + self.log_var_cls
        if reg_term is not None:
            precision_reg = torch.exp(-self.log_var_reg)
            loss = loss + precision_reg * reg_term + self.log_var_reg
        return loss


def build_regression_loss(loss_type, huber_beta):
    if loss_type == 'smoothl1':
        return SmoothL1Loss(beta=huber_beta).cuda()
    return MSELoss().cuda()


def choose_regression_loss(epoch, reg_loss_stage1, reg_loss_stage2, stage2_start_epoch):
    if reg_loss_stage2 is None:
        return reg_loss_stage1
    if epoch >= stage2_start_epoch:
        return reg_loss_stage2
    return reg_loss_stage1


def fit_linear_calibration(preds, labels, eps=1e-8):
    x = np.asarray(preds).reshape(-1).astype(np.float64)
    y = np.asarray(labels).reshape(-1).astype(np.float64)
    if x.size == 0 or y.size == 0:
        return 1.0, 0.0

    var_x = np.var(x)
    if var_x < eps:
        return 1.0, 0.0

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    coef = cov_xy / (var_x + eps)
    bias = mean_y - coef * mean_x
    return float(coef), float(bias)


def apply_linear_calibration(preds, coef, bias, clip_abs=None):
    preds_np = np.asarray(preds)
    shape = preds_np.shape
    calibrated = coef * preds_np.reshape(-1).astype(np.float64) + bias
    if clip_abs is not None and clip_abs > 0:
        calibrated = np.clip(calibrated, -clip_abs, clip_abs)
    return calibrated.reshape(shape).astype(np.float32)


def recalibrate_results(results, dataloader_class, coef, bias, clip_abs=None):
    if 'valpreds' not in results or 'vallabels' not in results:
        return results

    calibrated_preds = apply_linear_calibration(
        results['valpreds'],
        coef=coef,
        bias=bias,
        clip_abs=clip_abs,
    )
    recalculated, _ = dataloader_class.calculate_results(
        results.get('emoprobs', []),
        results.get('emolabels', []),
        calibrated_preds,
        results['vallabels'],
    )
    merged = dict(results)
    merged.update(recalculated)
    return merged


def fit_emotion_group_calibration(emo_probs, val_preds, val_labels, min_samples=20):
    probs = np.asarray(emo_probs)
    preds = np.asarray(val_preds).reshape(-1).astype(np.float64)
    labels = np.asarray(val_labels).reshape(-1).astype(np.float64)
    if probs.size == 0 or preds.size == 0 or labels.size == 0:
        return None

    pred_emos = np.argmax(probs, axis=1).astype(np.int64)
    global_coef, global_bias = fit_linear_calibration(preds, labels)
    groups = {}
    for emo in np.unique(pred_emos):
        idx = pred_emos == emo
        if np.sum(idx) < int(min_samples):
            continue
        coef, bias = fit_linear_calibration(preds[idx], labels[idx])
        groups[int(emo)] = (float(coef), float(bias))
    return {
        'global_coef': float(global_coef),
        'global_bias': float(global_bias),
        'groups': groups,
    }


def apply_emotion_group_calibration(emo_probs, val_preds, calibration, clip_abs=None):
    probs = np.asarray(emo_probs)
    preds_np = np.asarray(val_preds)
    flat_preds = preds_np.reshape(-1).astype(np.float64)
    shape = preds_np.shape

    if calibration is None or probs.size == 0:
        calibrated = flat_preds
    else:
        pred_emos = np.argmax(probs, axis=1).astype(np.int64)
        calibrated = calibration['global_coef'] * flat_preds + calibration['global_bias']
        for emo, (coef, bias) in calibration.get('groups', {}).items():
            idx = pred_emos == int(emo)
            if np.any(idx):
                calibrated[idx] = coef * flat_preds[idx] + bias

    if clip_abs is not None and clip_abs > 0:
        calibrated = np.clip(calibrated, -clip_abs, clip_abs)
    return calibrated.reshape(shape).astype(np.float32)


def recalibrate_results_by_emotion_group(results, dataloader_class, calibration, clip_abs=None):
    if ('valpreds' not in results) or ('vallabels' not in results):
        return results
    if ('emoprobs' not in results) or (len(results['emoprobs']) == 0):
        return results

    calibrated_preds = apply_emotion_group_calibration(
        emo_probs=results['emoprobs'],
        val_preds=results['valpreds'],
        calibration=calibration,
        clip_abs=clip_abs,
    )
    recalculated, _ = dataloader_class.calculate_results(
        results.get('emoprobs', []),
        results.get('emolabels', []),
        calibrated_preds,
        results['vallabels'],
    )
    merged = dict(results)
    merged.update(recalculated)
    return merged


def forward_with_tta(model, batch, tta_passes=1, tta_use_train_mode=False):
    if tta_passes <= 1:
        return model(batch)

    prev_mode = model.training
    if tta_use_train_mode:
        model.train()
    else:
        model.eval()

    feats, emos, vals = [], [], []
    with torch.no_grad():
        for _ in range(tta_passes):
            features, emos_out, vals_out, _ = model(batch)
            feats.append(features)
            emos.append(emos_out)
            vals.append(vals_out)

    if prev_mode:
        model.train()
    else:
        model.eval()

    feat_mean = torch.stack(feats, dim=0).mean(dim=0)
    emo_mean = torch.stack(emos, dim=0).mean(dim=0)
    val_mean = torch.stack(vals, dim=0).mean(dim=0)
    interloss = torch.tensor(0.0, device=feat_mean.device)
    return feat_mean, emo_mean, val_mean, interloss


def train_or_eval_regression_only(args, model, reg_loss, dataloader, epoch, optimizer=None, train=False):
    assert (not train) or (optimizer is not None)

    losses = []
    val_preds, val_labels = [], []
    config.train = train
    if train:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        batch, _, vals, _ = data
        for key in batch:
            batch[key] = batch[key].cuda(non_blocking=True)
        vals = vals.cuda(non_blocking=True)

        with torch.set_grad_enabled(train):
            _, _, vals_out, _ = model(batch)
            loss = reg_loss(vals_out, vals)

        if train:
            loss.backward()
            if model.model.grad_clip != -1:
                torch.nn.utils.clip_grad_value_(
                    [param for param in model.parameters() if param.requires_grad],
                    model.model.grad_clip,
                )
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        val_preds.append(vals_out.detach().cpu().numpy())
        val_labels.append(vals.detach().cpu().numpy())

    if len(val_preds) > 0:
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_mse = mean_squared_error(val_labels, val_preds)
    else:
        val_mse = float('inf')

    return {
        'loss': float(np.mean(losses)) if len(losses) > 0 else float('inf'),
        'valmse': float(val_mse),
        'num_samples': int(len(val_labels)) if isinstance(val_labels, np.ndarray) else 0,
    }


def finetune_regression_head(
    args,
    model,
    train_loader,
    eval_loader,
    reg_loss,
    fold_idx,
):
    if not args.use_reg_head_finetune:
        return 0, None
    if args.output_dim2 == 0:
        return 0, None
    if not hasattr(model, 'model'):
        return 0, None
    core_model = model.model
    if not hasattr(core_model, 'fc_out_2'):
        print(f'fold:{fold_idx+1}; reg-head finetune skipped (no fc_out_2)')
        return 0, None

    all_params = list(model.parameters())
    for p in all_params:
        p.requires_grad = False

    trainable = []
    for p in core_model.fc_out_2.parameters():
        p.requires_grad = True
        trainable.append(p)

    if getattr(args, 'use_valence_prior', False):
        if hasattr(core_model, 'valence_prior_gate'):
            core_model.valence_prior_gate.requires_grad = True
            trainable.append(core_model.valence_prior_gate)
        if hasattr(core_model, 'emo_valence_centers'):
            core_model.emo_valence_centers.requires_grad = True
            trainable.append(core_model.emo_valence_centers)

    if len(trainable) == 0:
        for p in all_params:
            p.requires_grad = True
        print(f'fold:{fold_idx+1}; reg-head finetune skipped (no trainable params)')
        return 0, None

    prev_use_modality_dropout = None
    if hasattr(core_model, 'use_modality_dropout'):
        prev_use_modality_dropout = bool(core_model.use_modality_dropout)
        core_model.use_modality_dropout = False

    optimizer = optim.AdamW(trainable, lr=args.reg_finetune_lr, weight_decay=args.l2)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=args.reg_finetune_lr_patience,
        verbose=False,
        min_lr=1e-6,
    )

    best_mse = float('inf')
    best_state = None
    no_improve = 0
    updates = 0
    for ft_epoch in range(args.reg_finetune_epochs):
        train_stats = train_or_eval_regression_only(
            args=args,
            model=model,
            reg_loss=reg_loss,
            dataloader=train_loader,
            epoch=ft_epoch,
            optimizer=optimizer,
            train=True,
        )
        eval_stats = train_or_eval_regression_only(
            args=args,
            model=model,
            reg_loss=reg_loss,
            dataloader=eval_loader,
            epoch=ft_epoch,
            optimizer=None,
            train=False,
        )
        scheduler.step(eval_stats['valmse'])
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'fold:{fold_idx+1}; reg-ft epoch:{ft_epoch+1}; '
            f'train_mse:{train_stats["valmse"]:.6f}; eval_mse:{eval_stats["valmse"]:.6f}; lr:{current_lr:.6f}'
        )

        if eval_stats['valmse'] < (best_mse - args.reg_finetune_min_delta):
            best_mse = eval_stats['valmse']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        updates += 1
        if no_improve >= args.reg_finetune_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if prev_use_modality_dropout is not None:
        core_model.use_modality_dropout = prev_use_modality_dropout
    for p in all_params:
        p.requires_grad = True

    return updates, best_mse if best_state is not None else None


def compute_fold_prior_centers_from_train_loader(train_loader, num_classes, ignore_missing_val_threshold=-9.0):
    dataset = getattr(train_loader, 'dataset', None)
    if dataset is None:
        return None, None
    raw_dataset = getattr(dataset, 'dataset', dataset)
    labels = getattr(raw_dataset, 'labels', None)
    if labels is None:
        return None, None

    sampler = getattr(train_loader, 'sampler', None)
    indices = getattr(sampler, 'indices', None)
    if indices is None:
        return None, None

    sums = np.zeros(num_classes, dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for idx in list(indices):
        item = labels[int(idx)]
        emo = int(item['emo'])
        val = float(item['val'])
        if val <= ignore_missing_val_threshold:
            continue
        if 0 <= emo < num_classes:
            sums[emo] += val
            counts[emo] += 1

    means = np.zeros(num_classes, dtype=np.float32)
    for emo in range(num_classes):
        if counts[emo] > 0:
            means[emo] = sums[emo] / counts[emo]
    return means, counts


def apply_fold_prior_centers_if_needed(args, model, train_loader, fold_idx):
    if not args.init_prior_from_fold_train:
        return
    if not hasattr(model, 'model'):
        return

    core_model = model.model
    if not hasattr(core_model, 'emo_valence_centers') or not hasattr(core_model, 'emo_center_init'):
        print(f'fold:{fold_idx+1}; prior-center init skipped (model has no emotion-valence centers)')
        return

    param = core_model.emo_valence_centers
    base = core_model.emo_center_init.detach().cpu().numpy().astype(np.float32).copy()
    num_classes = int(param.numel())
    means, counts = compute_fold_prior_centers_from_train_loader(
        train_loader=train_loader,
        num_classes=num_classes,
        ignore_missing_val_threshold=args.ignore_missing_val_threshold,
    )
    if means is None or counts is None:
        print(f'fold:{fold_idx+1}; prior-center init skipped (cannot access fold labels)')
        return

    centers = base
    for emo in range(num_classes):
        if counts[emo] > 0:
            centers[emo] = means[emo]

    centers_t = torch.tensor(centers, device=param.device, dtype=param.dtype)
    with torch.no_grad():
        core_model.emo_valence_centers.copy_(centers_t)
        core_model.emo_center_init.copy_(centers_t)

    counts_str = ",".join([str(int(x)) for x in counts.tolist()])
    centers_str = ",".join([f"{x:.3f}" for x in centers.tolist()])
    print(f'fold:{fold_idx+1}; init prior centers from fold-train; counts=[{counts_str}] centers=[{centers_str}]')


def train_or_eval_model(
    args,
    model,
    reg_loss,
    cls_loss,
    dataloader,
    dataloader_class,
    epoch,
    optimizer=None,
    train=False,
    task_weighter=None,
    inference_tta_passes=1,
    inference_tta_train_mode=False,
):
    vidnames = []
    val_preds, val_labels = [], []
    emo_probs, emo_labels = [], []
    losses = []

    assert (not train) or (optimizer is not None)
    config.train = train
    if train:
        model.train()
    else:
        model.eval()

    for iter, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        batch, emos, vals, bnames = data
        vidnames += bnames
        for key in batch:
            batch[key] = batch[key].cuda(non_blocking=True)
        emos = emos.cuda(non_blocking=True)
        vals = vals.cuda(non_blocking=True)

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            if train or inference_tta_passes <= 1:
                _, emos_out, vals_out, interloss = model(batch)
            else:
                _, emos_out, vals_out, interloss = forward_with_tta(
                    model=model,
                    batch=batch,
                    tta_passes=inference_tta_passes,
                    tta_use_train_mode=inference_tta_train_mode,
                )

            cls_term = None
            reg_term = None
            if args.output_dim1 != 0:
                cls_term = cls_loss(emos_out, emos)
                emo_probs.append(emos_out.detach().cpu().numpy())
                emo_labels.append(emos.detach().cpu().numpy())
            if args.output_dim2 != 0:
                reg_term = reg_loss(vals_out, vals)
                val_preds.append(vals_out.detach().cpu().numpy())
                val_labels.append(vals.detach().cpu().numpy())

            if args.use_uncertainty_weighted_mt and task_weighter is not None:
                loss = task_weighter(interloss, cls_term=cls_term, reg_term=reg_term)
            else:
                loss = interloss
                if cls_term is not None:
                    loss = loss + args.emo_loss_weight * cls_term
                if reg_term is not None:
                    loss = loss + args.val_loss_weight * reg_term
            losses.append(float(loss.detach().cpu().item()))

        if train:
            loss.backward()
            if model.model.grad_clip != -1:
                torch.nn.utils.clip_grad_value_(
                    [param for param in model.parameters() if param.requires_grad],
                    model.model.grad_clip,
                )
            optimizer.step()

        if (iter + 1) % args.print_iters == 0:
            print(f'process on {iter+1}|{len(dataloader)}, meanloss: {np.mean(losses)}')

    if emo_probs != []:
        emo_probs = np.concatenate(emo_probs)
    if emo_labels != []:
        emo_labels = np.concatenate(emo_labels)
    if val_preds != []:
        val_preds = np.concatenate(val_preds)
    if val_labels != []:
        val_labels = np.concatenate(val_labels)
    results, _ = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
    save_results = dict(
        names=vidnames,
        loss=np.mean(losses),
        **results,
    )
    return save_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Params for datasets
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--train_dataset', type=str, default=None, help='train dataset')
    parser.add_argument('--test_dataset',  type=str, default=None, help='test dataset')
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--savemodel', action='store_true', default=False, help='whether to save model')
    parser.add_argument('--save_iters', type=int, default=1e8, help='save models per iters')

    # Params for feature inputs
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature',  type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--feat_type',  type=str, default=None, help='feature type [utt, frm_align, frm_unalign]')
    parser.add_argument('--feat_scale', type=int, default=None, help='pre-compress input')
    parser.add_argument('--e2e_name', type=str, default=None, help='e2e pretrained model names')
    parser.add_argument('--e2e_dim',  type=int, default=None, help='e2e pretrained model hidden size')

    # Params for model
    parser.add_argument('--n_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--hyper_path', type=str, default=None, help='path to fixed hyperparams')
    parser.add_argument('--model', type=str, default='attention_robust', help='model name')

    # Params for training - 优化后的默认参数
    parser.add_argument('--lr', type=float, default=5e-4, metavar='lr', help='learning rate')
    parser.add_argument('--lr_adjust', type=str, default='case1', help='lr adjustment strategy')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2', help='L2 regularization weight (increased)')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--print_iters', type=int, default=1e8, help='print per-iteration')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--emo_loss_weight', type=float, default=1.0, help='classification loss weight')
    parser.add_argument('--val_loss_weight', type=float, default=1.0, help='regression loss weight')
    parser.add_argument('--reg_loss_type', type=str, default='mse', choices=['mse', 'smoothl1'], help='regression loss type')
    parser.add_argument('--huber_beta', type=float, default=1.0, help='beta for smoothl1 regression loss')
    parser.add_argument('--reg_loss_type_stage2', type=str, default='none', choices=['none', 'mse', 'smoothl1'], help='second-stage regression loss type')
    parser.add_argument('--reg_stage2_start_epoch', type=int, default=28, help='epoch to switch to second-stage regression loss')
    parser.add_argument('--huber_beta_stage2', type=float, default=0.8, help='beta for stage2 smoothl1 loss')
    parser.add_argument('--use_uncertainty_weighted_mt', action='store_true', default=False, help='use uncertainty-weighted multi-task losses')
    parser.add_argument('--mt_init_log_var_cls', type=float, default=0.0, help='init log-variance for cls task')
    parser.add_argument('--mt_init_log_var_reg', type=float, default=0.0, help='init log-variance for reg task')
    parser.add_argument('--use_swa', action='store_true', default=False, help='enable SWA in late training')
    parser.add_argument('--swa_start_epoch', type=int, default=30, help='start epoch for SWA averaging')
    parser.add_argument('--swa_lr', type=float, default=2e-4, help='SWA learning rate')
    parser.add_argument('--use_valence_calibration', action='store_true', default=False, help='fit linear calibration on eval valence and apply to test valence')
    parser.add_argument('--no_valence_calibration', action='store_false', dest='use_valence_calibration', help='disable valence calibration')
    parser.add_argument('--use_emotion_group_calibration', action='store_true', default=False, help='fit per-predicted-emotion calibration for valence')
    parser.add_argument('--emotion_group_calibration_min_samples', type=int, default=20, help='minimum eval samples required per emotion group')
    parser.add_argument('--valence_calibration_clip', type=float, default=-1.0, help='if >0, clip calibrated valence to [-clip, clip]')
    parser.add_argument('--use_reg_head_finetune', action='store_true', default=False, help='finetune regression head after base training')
    parser.add_argument('--reg_finetune_epochs', type=int, default=8, help='max epochs for regression-head finetune')
    parser.add_argument('--reg_finetune_lr', type=float, default=1e-4, help='learning rate for regression-head finetune')
    parser.add_argument('--reg_finetune_patience', type=int, default=3, help='early-stop patience for regression-head finetune')
    parser.add_argument('--reg_finetune_lr_patience', type=int, default=2, help='lr scheduler patience for regression-head finetune')
    parser.add_argument('--reg_finetune_min_delta', type=float, default=1e-4, help='min improvement for regression-head finetune')
    parser.add_argument('--tta_passes', type=int, default=1, help='number of TTA passes for final eval/test')
    parser.add_argument('--tta_use_train_mode', action='store_true', default=False, help='enable dropout-style TTA using train mode')
    parser.add_argument('--test_each_epoch', action='store_true', default=False, help='evaluate test sets every epoch (slower)')
    
    # 新增参数 - 针对模态缺失优化
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate (increased for regularization)')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping value')
    parser.add_argument('--modality_dropout', type=float, default=0.3, help='modality dropout rate for robustness')
    parser.add_argument('--use_modality_dropout', action='store_true', default=True, help='whether to use modality dropout')
    parser.add_argument('--no_modality_dropout', action='store_false', dest='use_modality_dropout', help='disable modality dropout')
    parser.add_argument('--modality_dropout_warmup', type=int, default=0, help='warmup epochs before applying modality dropout')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=10, help='lr scheduler patience')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='lr reduction factor')
    
    # AttentionRobustV2专用参数 - VAE + 代理模态
    parser.add_argument('--use_vae', action='store_true', default=True, help='whether to use VAE encoder')
    parser.add_argument('--no_vae', action='store_false', dest='use_vae', help='disable VAE encoder')
    parser.add_argument('--kl_weight', type=float, default=0.01, help='KL divergence loss weight')
    parser.add_argument('--recon_weight', type=float, default=0.1, help='reconstruction loss weight')
    parser.add_argument('--cross_kl_weight', type=float, default=0.01, help='cross-modal KL loss weight')
    parser.add_argument('--use_proxy_attention', action='store_true', default=True, help='whether to use proxy cross-modal attention')
    parser.add_argument('--no_proxy_attention', action='store_false', dest='use_proxy_attention', help='disable proxy attention')
    parser.add_argument('--fusion_temperature', type=float, default=1.0, help='temperature for uncertainty weighted fusion')
    parser.add_argument('--num_attention_heads', type=int, default=4, help='number of attention heads for proxy attention')

    # V4新增参数: 对比学习
    parser.add_argument('--use_contrastive', action='store_true', default=True, help='whether to use contrastive learning')
    parser.add_argument('--no_contrastive', action='store_false', dest='use_contrastive', help='disable contrastive learning')
    parser.add_argument('--contrastive_weight', type=float, default=0.1, help='contrastive loss weight')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07, help='temperature for InfoNCE loss')

    # V4新增参数: 门控融合
    parser.add_argument('--use_gated_fusion', action='store_true', default=True, help='whether to use gated uncertainty fusion')
    parser.add_argument('--no_gated_fusion', action='store_false', dest='use_gated_fusion', help='disable gated fusion')
    parser.add_argument('--gate_alpha', type=float, default=0.5, help='balance between uncertainty and gate weights')

    # V4新增参数: Focal Loss
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='focal loss gamma parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor')

    # V5新增参数: 深度编码器 + Mixup + 动态KL
    parser.add_argument('--use_mixup', action='store_true', default=False, help='whether to use mixup augmentation')
    parser.add_argument('--no_mixup', action='store_false', dest='use_mixup', help='disable mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.4, help='mixup alpha parameter')
    parser.add_argument('--use_dynamic_kl', action='store_true', default=True, help='whether to use dynamic KL scheduling')
    parser.add_argument('--no_dynamic_kl', action='store_false', dest='use_dynamic_kl', help='disable dynamic KL')
    parser.add_argument('--kl_warmup_epochs', type=int, default=20, help='KL warmup epochs')

    # V7新增参数: Emotion-Valence一致性 + 噪声增强
    parser.add_argument('--use_valence_prior', action='store_true', default=False, help='whether to use emotion-guided valence prior')
    parser.add_argument('--no_valence_prior', action='store_false', dest='use_valence_prior', help='disable emotion-guided valence prior')
    parser.add_argument('--valence_consistency_weight', type=float, default=0.08, help='weight of valence consistency regularization')
    parser.add_argument('--valence_center_reg_weight', type=float, default=0.005, help='weight of emotion-valence center regularization')
    parser.add_argument('--init_prior_from_fold_train', action='store_true', default=False, help='initialize emotion-valence centers from current fold training labels')
    parser.add_argument('--no_init_prior_from_fold_train', action='store_false', dest='init_prior_from_fold_train', help='disable fold-based prior center initialization')
    parser.add_argument('--ignore_missing_val_threshold', type=float, default=-9.0, help='ignore labels with val <= threshold when estimating fold prior centers')
    parser.add_argument('--valence_prior_hidden_dim', type=int, default=64, help='hidden size for sample-wise prior gate MLP')
    parser.add_argument('--valence_prior_gate_dropout', type=float, default=0.1, help='dropout for sample-wise prior gate MLP')
    parser.add_argument('--feature_noise_std', type=float, default=0.02, help='std of feature noise augmentation')
    parser.add_argument('--feature_noise_prob', type=float, default=0.3, help='probability of applying feature noise')
    parser.add_argument('--feature_noise_warmup', type=int, default=10, help='warmup epochs before noise augmentation')

    # V8新增参数: 双路径融合 + 可靠度建模
    parser.add_argument('--use_gated_uncertainty', action='store_true', default=True, help='whether to use gated uncertainty fusion')
    parser.add_argument('--no_gated_uncertainty', action='store_false', dest='use_gated_uncertainty', help='disable gated uncertainty fusion')
    parser.add_argument('--fusion_residual_scale', type=float, default=0.4, help='residual branch contribution in dual-path fusion')
    parser.add_argument('--reliability_temperature', type=float, default=1.0, help='temperature for reliability weighting')
    parser.add_argument('--modality_agreement_weight', type=float, default=0.01, help='weight of modality agreement regularization')
    parser.add_argument('--weight_consistency_weight', type=float, default=0.02, help='weight of reliability/fusion weight consistency')
    parser.add_argument('--quality_weight', type=float, default=0.6, help='quality logit weight for quality-aware fusion')
    parser.add_argument('--impute_loss_weight', type=float, default=0.10, help='weight of cross-modal imputation loss')
    parser.add_argument('--consistency_emo_weight', type=float, default=0.08, help='weight of teacher-student emotion consistency')
    parser.add_argument('--consistency_val_weight', type=float, default=0.05, help='weight of teacher-student valence consistency')
    parser.add_argument('--corruption_max_rate', type=float, default=0.45, help='max modality corruption rate for training')
    parser.add_argument('--corruption_warmup_epochs', type=int, default=25, help='warmup epochs to reach corruption_max_rate')
    parser.add_argument('--double_mask_ratio', type=float, default=0.35, help='ratio of double-modality masking among corrupted samples')
    parser.add_argument('--latent_noise_std', type=float, default=0.02, help='std of latent noise for student branch')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)


    print ('====== Params Pre-analysis =======')
    if args.feat_type == 'utt':
        args.feat_scale = 1
    elif args.feat_type == 'frm_align':
        assert args.audio_feature.endswith('FRA')
        assert args.text_feature.endswith('FRA')
        assert args.video_feature.endswith('FRA')
        args.feat_scale = 6
    elif args.feat_type == 'frm_unalign':
        assert args.audio_feature.endswith('FRA')
        assert args.text_feature.endswith('FRA')
        assert args.video_feature.endswith('FRA')
        args.feat_scale = 12

    ## define store folder
    if args.train_dataset is not None:
        args.save_root = f'{args.save_root}-cross'
    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    whole_features = [item for item in whole_features if item is not None]
    if len(set(whole_features)) == 0:
        args.save_root = f'{args.save_root}-others'
    elif len(set(whole_features)) == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif len(set(whole_features)) == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif len(set(whole_features)) == 3:
        args.save_root = f'{args.save_root}-trimodal'

    config.dataset = args.dataset
    print('args: ', args)

    ## save root
    save_resroot  = os.path.join(args.save_root, 'result')
    save_modelroot  = os.path.join(args.save_root, 'model')
    if not os.path.exists(save_resroot):  os.makedirs(save_resroot)
    if not os.path.exists(save_modelroot): os.makedirs(save_modelroot)
    
    feature_name = "+".join(sorted(list(set(whole_features))))
    model_name = f'{args.model}+{args.feat_type}+{args.e2e_name}'
    prefix_name = f'features:{feature_name}_dataset:{args.dataset}_model:{model_name}'
    if args.train_dataset is not None:
        assert args.test_dataset is not None
        prefix_name += f'_train:{args.train_dataset}_test:{args.test_dataset}'


    print ('====== Reading Data =======')
    dataloader_class = get_dataloaders(args)
    train_loaders, eval_loaders, test_loaders = dataloader_class.get_loaders()
    assert len(train_loaders) == len(eval_loaders)
    print (f'train&val folder:{len(train_loaders)}; test sets:{len(test_loaders)}')
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()


    print ('====== Training and Evaluation =======')
    folder_save = []
    folder_duration = []
    for ii in range(len(train_loaders)):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        eval_loader  = eval_loaders[ii]
        start_time = name_time = time.time()

        print (f'Step1: build model (each folder has its own model)')
        model = get_models(args).cuda()
        apply_fold_prior_centers_if_needed(args=args, model=model, train_loader=train_loader, fold_idx=ii)
        task_weighter = None
        extra_params = []
        if args.use_uncertainty_weighted_mt:
            task_weighter = UncertaintyWeightedTaskLoss(
                init_log_var_cls=args.mt_init_log_var_cls,
                init_log_var_reg=args.mt_init_log_var_reg,
            ).cuda()
            extra_params.extend(list(task_weighter.parameters()))
        reg_loss_stage1 = build_regression_loss(args.reg_loss_type, args.huber_beta)
        reg_loss_stage2 = None
        if args.reg_loss_type_stage2 != 'none':
            stage2_beta = args.huber_beta_stage2 if args.reg_loss_type_stage2 == 'smoothl1' else args.huber_beta
            reg_loss_stage2 = build_regression_loss(args.reg_loss_type_stage2, stage2_beta)
        cls_loss = CELoss().cuda()

        trainable_params = list(model.parameters()) + extra_params
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.l2)
        
        # 学习率调度器 - 当验证指标不再提升时降低学习率
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',  # 最大化验证指标
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=True,
            min_lr=1e-6
        )
        
        # 早停机制
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience, 
            min_delta=0.001, 
            mode='max'
        )
        
        swa_model = None
        swa_scheduler = None
        swa_updates = 0
        if args.use_swa:
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)

        print (f'Step2: training (multiple epoches)')
        whole_metrics = []
        best_model_state = None
        best_eval_results = None
        best_epoch = 0
        
        for epoch in range(args.epochs):
            epoch_store = {}
            
            # 设置当前epoch用于渐进式模态dropout
            if hasattr(model.model, 'set_epoch'):
                model.model.set_epoch(epoch)

            epoch_reg_loss = choose_regression_loss(
                epoch=epoch,
                reg_loss_stage1=reg_loss_stage1,
                reg_loss_stage2=reg_loss_stage2,
                stage2_start_epoch=args.reg_stage2_start_epoch,
            )
            reg_loss_tag = args.reg_loss_type_stage2 if (
                reg_loss_stage2 is not None and epoch >= args.reg_stage2_start_epoch
            ) else args.reg_loss_type

            train_results = train_or_eval_model(args, model, epoch_reg_loss, cls_loss, train_loader, dataloader_class, epoch=epoch, optimizer=optimizer, train=True, task_weighter=task_weighter)
            eval_results  = train_or_eval_model(args, model, epoch_reg_loss, cls_loss, eval_loader, dataloader_class, epoch=epoch, optimizer=None,      train=False, task_weighter=task_weighter)
            func_update_storage(inputs=eval_results, prefix='eval', outputs=epoch_store)

            train_metric = gain_metric_from_results(train_results, args.metric_name)
            eval_metric  = gain_metric_from_results(eval_results,  args.metric_name)
            whole_metrics.append(eval_metric)
            
            # 更新学习率调度器 / SWA调度器
            if args.use_swa and epoch >= args.swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
                swa_updates += 1
            else:
                scheduler.step(eval_metric)
            current_lr = optimizer.param_groups[0]['lr']
            
            print (f'epoch:{epoch+1}; metric:{args.metric_name}; train:{train_metric:.4f}; eval:{eval_metric:.4f}; lr:{current_lr:.6f}; reg_loss:{reg_loss_tag}')

            if args.test_each_epoch:
                for jj, test_loader in enumerate(test_loaders):
                    test_results = train_or_eval_model(args, model, epoch_reg_loss, cls_loss, test_loader, dataloader_class, epoch=epoch, optimizer=None, train=False, task_weighter=task_weighter)
                    func_update_storage(inputs=test_results, prefix=f'test{jj+1}', outputs=epoch_store)
            
            # 保存最佳模型状态
            if eval_metric >= max(whole_metrics):
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_eval_results = eval_results
                best_epoch = epoch

            # 检查早停
            if early_stopping(eval_metric, epoch):
                print(f'Early stopping at epoch {epoch+1}, best epoch: {early_stopping.best_epoch+1}')
                break

        print (f'Step3: saving and testing on the {ii+1} folder')
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        best_epoch_reg_loss = choose_regression_loss(
            epoch=best_epoch,
            reg_loss_stage1=reg_loss_stage1,
            reg_loss_stage2=reg_loss_stage2,
            stage2_start_epoch=args.reg_stage2_start_epoch,
        )

        reg_ft_updates = 0
        reg_ft_best_mse = None
        if args.use_reg_head_finetune:
            if args.use_swa and swa_updates > 0:
                print(f'fold:{ii+1}; reg-head finetune skipped because SWA model is active')
            else:
                reg_ft_updates, reg_ft_best_mse = finetune_regression_head(
                    args=args,
                    model=model,
                    train_loader=train_loader,
                    eval_loader=eval_loader,
                    reg_loss=best_epoch_reg_loss,
                    fold_idx=ii,
                )
                print(
                    f'fold:{ii+1}; reg-head finetune done; updates:{reg_ft_updates}; '
                    f'best_eval_mse:{reg_ft_best_mse}'
                )

        eval_model = swa_model if (args.use_swa and swa_updates > 0 and (not args.use_reg_head_finetune)) else model
        final_eval_tta_passes = max(1, int(args.tta_passes))
        best_eval_results = train_or_eval_model(
            args,
            eval_model,
            best_epoch_reg_loss,
            cls_loss,
            eval_loader,
            dataloader_class,
            epoch=best_epoch,
            optimizer=None,
            train=False,
            task_weighter=task_weighter,
            inference_tta_passes=final_eval_tta_passes,
            inference_tta_train_mode=args.tta_use_train_mode,
        )

        calibration_coef = None
        calibration_bias = None
        group_calibration = None
        if (
            args.use_emotion_group_calibration
            and args.output_dim2 != 0
            and ('valpreds' in best_eval_results)
            and ('vallabels' in best_eval_results)
            and ('emoprobs' in best_eval_results)
        ):
            group_calibration = fit_emotion_group_calibration(
                emo_probs=best_eval_results['emoprobs'],
                val_preds=best_eval_results['valpreds'],
                val_labels=best_eval_results['vallabels'],
                min_samples=args.emotion_group_calibration_min_samples,
            )
            if group_calibration is not None:
                best_eval_results = recalibrate_results_by_emotion_group(
                    best_eval_results,
                    dataloader_class=dataloader_class,
                    calibration=group_calibration,
                    clip_abs=args.valence_calibration_clip,
                )
                print(
                    f'fold:{ii+1}; emotion-group calibration enabled; '
                    f'groups={sorted(list(group_calibration.get("groups", {}).keys()))}; '
                    f'global_coef:{group_calibration["global_coef"]:.6f}; '
                    f'global_bias:{group_calibration["global_bias"]:.6f}; '
                    f'clip:{args.valence_calibration_clip}'
                )
        elif args.use_valence_calibration and args.output_dim2 != 0 and ('valpreds' in best_eval_results) and ('vallabels' in best_eval_results):
            calibration_coef, calibration_bias = fit_linear_calibration(
                best_eval_results['valpreds'],
                best_eval_results['vallabels'],
            )
            best_eval_results = recalibrate_results(
                best_eval_results,
                dataloader_class=dataloader_class,
                coef=calibration_coef,
                bias=calibration_bias,
                clip_abs=args.valence_calibration_clip,
            )
            print(
                f'fold:{ii+1}; valence calibration enabled; coef:{calibration_coef:.6f}; '
                f'bias:{calibration_bias:.6f}; clip:{args.valence_calibration_clip}'
            )

        best_store = {}
        if calibration_coef is not None:
            best_store['calibration_coef'] = calibration_coef
            best_store['calibration_bias'] = calibration_bias
        if group_calibration is not None:
            best_store['group_calibration_global_coef'] = group_calibration['global_coef']
            best_store['group_calibration_global_bias'] = group_calibration['global_bias']
            best_store['group_calibration_num_groups'] = len(group_calibration.get('groups', {}))
        if args.use_reg_head_finetune:
            best_store['reg_ft_updates'] = reg_ft_updates
            best_store['reg_ft_best_mse'] = -1.0 if reg_ft_best_mse is None else reg_ft_best_mse
        func_update_storage(inputs=best_eval_results, prefix='eval', outputs=best_store)

        for jj, test_loader in enumerate(test_loaders):
            test_results = train_or_eval_model(
                args,
                eval_model,
                best_epoch_reg_loss,
                cls_loss,
                test_loader,
                dataloader_class,
                epoch=best_epoch,
                optimizer=None,
                train=False,
                task_weighter=task_weighter,
                inference_tta_passes=final_eval_tta_passes,
                inference_tta_train_mode=args.tta_use_train_mode,
            )
            if group_calibration is not None:
                test_results = recalibrate_results_by_emotion_group(
                    test_results,
                    dataloader_class=dataloader_class,
                    calibration=group_calibration,
                    clip_abs=args.valence_calibration_clip,
                )
            elif calibration_coef is not None:
                test_results = recalibrate_results(
                    test_results,
                    dataloader_class=dataloader_class,
                    coef=calibration_coef,
                    bias=calibration_bias,
                    clip_abs=args.valence_calibration_clip,
                )
            func_update_storage(inputs=test_results, prefix=f'test{jj+1}', outputs=best_store)
        folder_save.append(best_store)
        end_time = time.time()
        duration = end_time - start_time
        folder_duration.append(duration)
        print (f'>>>>> Finish: training on the {ii+1}-th folder, best_epoch: {best_epoch+1}, swa_updates: {swa_updates}, duration: {duration} >>>>>')
        
        del model
        del task_weighter
        del optimizer
        torch.cuda.empty_cache()


    print ('====== Prediction and Saving =======')
    args.duration = np.sum(folder_duration)
    cv_result = gain_cv_results(folder_save)
    save_path = f'{save_resroot}/cv_{prefix_name}_{cv_result}_{name_time}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path, args=np.array(args, dtype=object))

    for jj in range(len(test_loaders)):
        emo_labels, emo_probs = average_folder_for_emos(folder_save, f'test{jj+1}')
        val_labels, val_preds = average_folder_for_vals(folder_save, f'test{jj+1}')
        _, test_result = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
        save_path = f'{save_resroot}/test{jj+1}_{prefix_name}_{test_result}_{name_time}.npz'
        print (f'save results in {save_path}')
        np.savez_compressed(save_path, args=np.array(args, dtype=object))
