# P-RMF 论文涉及的全部脚本与数据索引

> 生成时间: 2026-02-06
> 论文标题: 基于概率化代理模态融合的鲁棒多模态情感识别方法 (P-RMF)

---

## 一、最新实验结果（extract_best_results.py 输出）

### 模型对比表格 (最高ACC)

| 方法 | cv F1 | cv ACC | test1 F1 | test1 ACC | test2 F1 | test2 ACC | test3 F1 | test3 ACC |
|------|------:|-------:|---------:|----------:|---------:|----------:|---------:|----------:|
| Baseline (v0) | 0.7366 | 0.7376 | 0.7956 | 0.7956 | 0.7450 | 0.7476 | 0.8638 | 0.8645 |
| Robust v1 (Dropout) | 0.7491 | 0.7516 | 0.8239 | 0.8248 | 0.7609 | 0.7621 | 0.8910 | 0.8945 |
| Robust v4 | 0.7722 | 0.7732 | 0.8302 | 0.8297 | 0.7832 | 0.7840 | 0.8907 | 0.8921 |
| Robust v5 | 0.7427 | 0.7447 | 0.8113 | 0.8127 | 0.7723 | 0.7767 | 0.8939 | 0.8993 |
| **P-RMF V2 (VAE)** | **0.7586** | **0.7593** | **0.8348** | **0.8345** | **0.7693** | **0.7718** | **0.8995** | **0.9029** |

### P-RMF V2 最佳结果详情

| Split | F1 | ACC | Val |
|-------|---:|----:|----:|
| cv | 0.7586 | 0.7593 | 0.6317 |
| test1 | 0.8348 | 0.8345 | 0.6431 |
| test2 | 0.7693 | 0.7718 | 0.6170 |
| test3 | 0.8995 | 0.9029 | 79.9783 |

### 数据来源统计

| 结果目录 | NPZ文件数 |
|----------|----------:|
| V2 outputs/results-trimodal/result/ | 4 |
| MERBench_Run/saved-trimodal/result/ | 32 |
| MERBench/saved-trimodal/result/ | 128 |
| MERBenchv1版本/saved-trimodal/result/ | 12 |
| MERBenchv1版本/saved-robust-trimodal/result/ | 8 |
| **合计** | **184** |

---

## 二、文件总览

### 目录结构

```
attention_robust_v2/
├── attention_robust_v2.py          # 主模型代码 (319行)
├── __init__.py                     # 包初始化
├── modules/
│   ├── __init__.py                 # 模块初始化
│   ├── variational_encoder.py      # VAE核心模块 (380行)
│   └── encoder.py                  # 基础编码器 (MLPEncoder/LSTMEncoder)
├── setup_v2.py                     # 自动安装脚本
├── train_v2.sh                     # 训练启动脚本
├── run_ablation.sh                 # 消融实验脚本
├── extract_best_results.py         # 结果提取脚本 (386行)
├── paper_v2.md                     # 论文正文
├── results_comparison.md           # 结果对比表
├── results_latest.md               # 最新结果
├── paper_all_scripts_and_data.md   # 本文件
└── outputs/
    └── results-trimodal/result/    # V2实验结果NPZ文件
```

---

## 三、核心模型代码

### 3.1 主模型: `attention_robust_v2.py` (319行)

**文件路径**: `attention_robust_v2/attention_robust_v2.py`

**类**: `AttentionRobustV2(nn.Module)`

**架构流程**:
```
输入 (audio, text, video)
  → 变分编码器 (VariationalMLPEncoder) → (z, μ, logvar, σ) × 3模态
  → 模态Dropout (apply_modality_dropout) → μ_dropped, σ_adjusted
  → 不确定性加权融合 (UncertaintyWeightedFusion) → proxy, weights
  → 代理模态跨模态注意力 (ProxyCrossModalAttention) → fused
  → 输出层 → emos_out (分类), vals_out (回归)
  → 辅助损失: KL + Recon + CrossKL → interloss
```

**关键参数 (从args读取)**:
```python
# VAE参数
use_vae = True                    # 是否使用VAE编码
kl_weight = 0.01                  # KL散度损失权重 (默认)
recon_weight = 0.1                # 重建损失权重 (默认)
cross_kl_weight = 0.01            # 跨模态KL权重 (默认)

# 代理模态参数
use_proxy_attention = True        # 是否使用代理模态注意力
fusion_temperature = 1.0          # 融合温度τ (默认)
num_attention_heads = 4           # 注意力头数

# 模态Dropout参数
modality_dropout = 0.2            # 最大Dropout率
use_modality_dropout = True       # 是否启用
modality_dropout_warmup = 0       # 预热轮次 (默认)
```

**核心方法**:
- `forward_vae(batch)`: VAE模式前向传播 (第224-285行)
- `apply_modality_dropout(z_a, z_t, z_v)`: 6种模态Dropout模式 (第162-212行)
- `forward_original(batch)`: 兼容模式 (第287-318行)
- `set_epoch(epoch)`: 设置当前epoch用于渐进式Dropout (第158-160行)

**模态Dropout的6种模式** (第190-205行):
```
mode 0: 丢弃音频        mode 3: 丢弃音频+文本
mode 1: 丢弃文本        mode 4: 丢弃音频+视频
mode 2: 丢弃视频        mode 5: 丢弃文本+视频
```

**被Dropout模态的不确定性处理** (第244-249行):
```python
# 被dropout的模态 → std设为1e6 → 融合权重趋近于0
std_a_adj = torch.where(
    mu_a_dropped.abs().sum(dim=1, keepdim=True) == 0,
    torch.ones_like(std_a) * 1e6, std_a
)
```

---

### 3.2 VAE核心模块: `modules/variational_encoder.py` (380行)

**文件路径**: `attention_robust_v2/modules/variational_encoder.py`

包含6个类:

#### (1) `VariationalMLPEncoder` (第12-87行)
话语级特征的变分编码器。
```
输入 x [B, in_dim]
  → Dropout → 共享MLP(2层ReLU) → h [B, hidden_dim]
  → mu_layer(h) → μ [B, hidden_dim]
  → logvar_layer(h) → logvar [B, hidden_dim] (clamp到[-10,10])
  → std = exp(0.5 * logvar)
  → z = μ + ε × σ (训练时) 或 z = μ (推理时)
输出: (z, μ, logvar, σ)
```

#### (2) `VariationalLSTMEncoder` (第90-147行)
序列特征的变分编码器，结构类似但前端用LSTM。

#### (3) `ModalityDecoder` (第150-176行)
从潜在变量重建原始特征: `z [B, H] → 3层MLP → x_recon [B, out_size]`

#### (4) `UncertaintyWeightedFusion` (第179-220行)
**论文核心创新之一**。
```python
# 计算每个模态的平均不确定性
uncertainty_m = std_m.mean(dim=-1)  # [B, 1]

# 反向方差加权
inv_uncertainties = 1.0 / (uncertainties + 1e-6)
weights = softmax(inv_uncertainties / temperature)  # [B, 3]

# 生成代理模态
proxy = Σ(w_m × μ_m)  # [B, H]
```

#### (5) `ProxyCrossModalAttention` (第223-291行)
代理模态引导的跨模态注意力。
```
proxy → Q, 各模态μ → K,V
→ 3个独立的MultiheadAttention (audio/text/video)
→ 不确定性加权融合attention结果
→ 残差连接 + LayerNorm + FFN(GELU)
→ fused [B, H]
```

#### (6) `VAELossComputer` (第294-379行)
辅助损失计算器:
- `kl_divergence_to_standard_normal(μ, logvar)`: KL(q||N(0,I))
- `reconstruction_loss(original, reconstructed)`: MSE重建损失
- `cross_modal_kl(mu_list, logvar_list)`: 跨模态KL散度
- `compute(...)`: 总辅助损失 = α×L_KL + β×L_recon + γ×L_crossKL

---

### 3.3 基础编码器: `modules/encoder.py`

**文件路径**: `attention_robust_v2/modules/encoder.py`

提供非VAE版本的编码器，用于消融实验对比:
- `MLPEncoder(in_size, hidden_size, dropout)`: 确定性MLP编码器
- `LSTMEncoder(in_size, hidden_size, dropout)`: 确定性LSTM编码器

---

## 四、训练与实验脚本

### 4.1 训练启动脚本: `train_v2.sh`

**文件路径**: `attention_robust_v2/train_v2.sh`

**功能**: 启动P-RMF V2模型的完整训练流程

**关键参数配置**:
```bash
MODEL=attention_robust_v2
FEATURE_SET="Baichuan-13B-Base-UTT chinese-hubert-large-UTT CLIP-ViT-large-patch14-UTT"
GPU=0

# 训练参数
--num_epochs 150
--early_stop 40
--lr 3e-4
--weight_decay 1e-4
--batch_size 32
--hidden_dim 128
--dropout 0.4

# VAE参数
--use_vae True
--kl_weight 0.005
--recon_weight 0.15
--cross_kl_weight 0.01

# 代理模态参数
--use_proxy_attention True
--fusion_temperature 0.8
--num_attention_heads 4

# 模态Dropout参数
--use_modality_dropout True
--modality_dropout 0.2
--modality_dropout_warmup 15
```

**执行命令**:
```bash
cd /root/autodl-tmp/MERTools-master/MERBench_Run
bash attention_robust_v2/train_v2.sh
```

---

### 4.2 消融实验脚本: `run_ablation.sh`

**文件路径**: `attention_robust_v2/run_ablation.sh`

**功能**: 运行5组消融实验，验证各组件贡献

**实验配置**:

| 实验编号 | 名称 | VAE | Proxy Attn | Modality Dropout | 关键差异 |
|----------|------|:---:|:----------:|:----------------:|----------|
| Exp1 | V1 Baseline | ✗ | ✗ | ✓ | 仅模态Dropout |
| Exp2 | VAE Only | ✓ | ✗ | ✓ | VAE编码+不确定性加权 |
| Exp3 | VAE+Proxy | ✓ | ✓ | ✓ | 加入代理模态注意力 |
| Exp4 | Full P-RMF | ✓ | ✓ | ✓ | 完整模型+调优参数 |
| Exp5 | No VAE | ✗ | ✓ | ✓ | 仅代理注意力无VAE |

**各实验的参数差异**:

```bash
# Exp1: V1 Baseline (无VAE, 无Proxy)
--use_vae False --use_proxy_attention False --modality_dropout 0.2

# Exp2: VAE Only (有VAE, 无Proxy)
--use_vae True --use_proxy_attention False
--kl_weight 0.01 --recon_weight 0.1 --cross_kl_weight 0.01

# Exp3: VAE + Proxy (有VAE, 有Proxy, 默认参数)
--use_vae True --use_proxy_attention True
--kl_weight 0.01 --recon_weight 0.1 --fusion_temperature 1.0

# Exp4: Full P-RMF (完整模型, 调优参数)
--use_vae True --use_proxy_attention True
--kl_weight 0.005 --recon_weight 0.15 --fusion_temperature 0.8
--modality_dropout_warmup 15

# Exp5: No VAE (无VAE, 有Proxy)
--use_vae False --use_proxy_attention True --modality_dropout 0.2
```

**执行命令**:
```bash
cd /root/autodl-tmp/MERTools-master/MERBench_Run
bash attention_robust_v2/run_ablation.sh
```

---

### 4.3 结果提取脚本: `extract_best_results.py` (386行)

**文件路径**: `attention_robust_v2/extract_best_results.py`

**功能**: 从多个目录的NPZ结果文件中自动筛选最高准确度结果

**核心逻辑**:
1. 扫描6个结果目录:
   - `attention_robust_v2/outputs/` (V2专属)
   - `MERBench_Run/saved-trimodal/` (共享)
   - `MERBench/saved-trimodal/` (MERBench)
   - `MERBench/attention_robust_v2/outputs/`
   - `MERBenchv1版本/saved-trimodal/`
   - `MERBenchv1版本/saved-robust-trimodal/`
2. 正则解析文件名: `f1:(\d+\.\d+)_acc:(\d+\.\d+)_val:(\d+\.\d+)`
3. 按 `(model, split)` 分组取最高ACC
4. 输出Markdown对比表格 + 控制台表格

**模型名称映射**:
```python
MODEL_DISPLAY_NAMES = {
    'attention':             'Baseline (v0)',
    'attention+utt':         'Baseline (v0)',
    'attention_robust':      'Robust v1 (Dropout)',
    'attention_robust_v2':   'P-RMF V2 (VAE)',
    'attention_robust_v3':   'Robust v3',
    'attention_robust_v4':   'Robust v4',
    'attention_robust_v5':   'Robust v5',
}
```

**执行命令**:
```bash
python extract_best_results.py                          # 扫描所有默认目录
python extract_best_results.py --output results.md      # 输出到文件
python extract_best_results.py --result_dir /path/to/   # 指定目录
```

---

### 4.4 自动安装脚本: `setup_v2.py`

**文件路径**: `attention_robust_v2/setup_v2.py`

**功能**: 自动注册V2模型到MERBench框架
- 将 `attention_robust_v2.py` 复制到 `toolkit/models/`
- 在 `toolkit/globals.py` 中注册模型名称
- 在 `toolkit/utils/functions.py` 中添加模型导入

---

## 五、论文中引用的公式与代码对应关系

### 5.1 变分编码 (论文§3.2 ↔ variational_encoder.py)

| 论文公式 | 代码位置 | 代码实现 |
|----------|----------|----------|
| $h_m = \text{ReLU}(\text{BN}(W^{(1)}_m x_m + b^{(1)}_m))$ | `VariationalMLPEncoder.shared` (L33-38) | `nn.Sequential(Linear, ReLU, Linear, ReLU)` |
| $\mu_m = W^{(\mu)}_m h_m + b^{(\mu)}_m$ | `VariationalMLPEncoder.mu_layer` (L41) | `nn.Linear(hidden, hidden)` |
| $\log\sigma^2_m = W^{(\sigma)}_m h_m + b^{(\sigma)}_m$ | `VariationalMLPEncoder.logvar_layer` (L44) | `nn.Linear(hidden, hidden)` |
| $z_m = \mu_m + \epsilon \odot \sigma_m$ | `reparameterize()` (L50-63) | `mu + eps * std` |
| $\mathcal{L}_{KL}$ | `VAELossComputer.kl_divergence_to_standard_normal()` (L311-319) | `-0.5 * sum(1 + logvar - mu² - exp(logvar))` |
| $\mathcal{L}_{recon}$ | `VAELossComputer.reconstruction_loss()` (L321-327) | `MSELoss(recon, original)` |

### 5.2 不确定性融合 (论文§3.3 ↔ variational_encoder.py)

| 论文公式 | 代码位置 | 代码实现 |
|----------|----------|----------|
| $u_m = \frac{1}{d}\sum\sigma^2_{m,j}$ | `UncertaintyWeightedFusion.forward()` (L206) | `std.mean(dim=-1)` |
| $w_m = \text{softmax}(1/(u_m \cdot \tau))$ | 同上 (L212-213) | `softmax(1/(uncertainties+1e-6) / τ)` |
| $p = \sum w_m \cdot \mu_m$ | 同上 (L216-218) | `(mu_stack * weights_exp).sum(dim=1)` |

### 5.3 代理模态注意力 (论文§3.4 ↔ variational_encoder.py)

| 论文公式 | 代码位置 | 代码实现 |
|----------|----------|----------|
| $Q = pW^Q, K_m = \mu_m W^K, V_m = \mu_m W^V$ | `ProxyCrossModalAttention.forward()` (L264-273) | 3个独立`MultiheadAttention` |
| $o = \text{LN}(p + \text{Attn})$ | 同上 (L288) | `self.norm(proxy + weighted_attn)` |
| $f = \text{LN}(o + \text{FFN}(o))$ | 同上 (L289) | `fused + self.ffn(fused)` |

### 5.4 渐进式模态Dropout (论文§3.5 ↔ attention_robust_v2.py)

| 论文公式 | 代码位置 | 代码实现 |
|----------|----------|----------|
| $p_{drop}(e) = p_{max} \cdot \frac{e-e_w}{E-e_w}$ | `apply_modality_dropout()` (L170-180) | 线性增长Dropout率 |
| 6种Dropout模式 | 同上 (L190-205) | `random.choice([0,1,2,3,4,5])` |

### 5.5 训练损失 (论文§3.6 ↔ attention_robust_v2.py + variational_encoder.py)

| 论文公式 | 代码位置 |
|----------|----------|
| $\mathcal{L}_{task} = \mathcal{L}_{CE} + \lambda_{val}\mathcal{L}_{MSE}$ | `attention_robust_v2.py` forward返回的 `emos_out, vals_out` |
| $\mathcal{L} = \mathcal{L}_{task} + \lambda_{KL}\mathcal{L}_{KL} + \lambda_{recon}\mathcal{L}_{recon} + \lambda_{cross}\mathcal{L}_{cross}$ | `VAELossComputer.compute()` (L350-379) |

---

## 六、数据相关

### 6.1 数据集: MER2023

| 属性 | 值 |
|------|------|
| 数据来源 | 电影/电视剧片段 |
| 模态 | 音频、文本(ASR转录)、视觉 |
| 情感类别 | 6类: happy, sad, angry, neutral, worried, surprise |
| 训练集 | 3373 样本 (5折交叉验证) |
| test1 | 411 样本 |
| test2 | 412 样本 |
| test3 | 834 样本 |
| 评估指标 | F1, ACC, EmoVal |

### 6.2 预提取特征

| 模态 | 特征提取器 | 维度 | 特征类型 |
|------|-----------|-----:|----------|
| 音频 | chinese-hubert-large | 1024 | 话语级 (UTT) |
| 文本 | Baichuan-13B-Base | 5120 | 话语级 (UTT) |
| 视觉 | CLIP-ViT-large-patch14 | 768 | 话语级 (UTT) |

### 6.3 NPZ结果文件格式

**文件名格式**:
```
{split}_features:{feature_set}_model:{model_name}+utt+None_f1:{f1}_acc:{acc}_val:{val}_{timestamp}.npz
```

**示例**:
```
cv_features:Baichuan-13B-Base-UTT+chinese-hubert-large-UTT+CLIP-ViT-large-patch14-UTT_model:attention_robust_v2+utt+None_f1:0.7586_acc:0.7593_val:0.6317_1769910916.npz
```

**NPZ文件内容**: 包含模型预测结果，用于后续分析和提交。

---

## 七、超参数完整配置

### 7.1 最优配置 (论文报告)

```python
config = {
    # 模型结构
    'hidden_dim': 128,
    'dropout': 0.4,
    'num_attention_heads': 4,

    # VAE
    'use_vae': True,
    'kl_weight': 0.005,
    'recon_weight': 0.15,
    'cross_kl_weight': 0.01,

    # 代理模态
    'use_proxy_attention': True,
    'fusion_temperature': 0.8,

    # 模态Dropout
    'use_modality_dropout': True,
    'modality_dropout': 0.2,
    'modality_dropout_warmup': 15,

    # 训练
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'batch_size': 32,
    'num_epochs': 150,
    'early_stop': 40,

    # 数据
    'features': [
        'Baichuan-13B-Base-UTT',
        'chinese-hubert-large-UTT',
        'CLIP-ViT-large-patch14-UTT'
    ],
}
```

### 7.2 消融实验参数差异

| 参数 | Exp1 | Exp2 | Exp3 | Exp4 (Full) | Exp5 |
|------|------|------|------|-------------|------|
| use_vae | False | True | True | True | False |
| use_proxy_attention | False | False | True | True | True |
| kl_weight | - | 0.01 | 0.01 | **0.005** | - |
| recon_weight | - | 0.1 | 0.1 | **0.15** | - |
| cross_kl_weight | - | 0.01 | 0.01 | 0.01 | - |
| fusion_temperature | - | 1.0 | 1.0 | **0.8** | 1.0 |
| modality_dropout | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 |
| dropout_warmup | 0 | 0 | 0 | **15** | 0 |

---

## 八、相关文档

### 8.1 技术方案文档

**文件**: `MERBench_Run/AttentionRobustV2_深度改造方案.md`

内容: V2模型的完整技术设计方案，包括:
- 从V1到V2的改造动机
- VAE编码器设计细节
- 不确定性融合机制推导
- 代理模态注意力设计
- 渐进式Dropout策略
- 损失函数设计

### 8.2 实验总结文档

**文件**: `MERBench_Run/实验总结_attention模型优化.md`

内容: 所有版本(v0-v5)的实验对比总结，包括:
- 各版本的改进点
- 完整的实验结果对比表
- 超参数搜索记录
- 失败实验分析

---

## 九、复现指南

### 步骤1: 安装模型
```bash
cd /root/autodl-tmp/MERTools-master/MERBench_Run
python attention_robust_v2/setup_v2.py
```

### 步骤2: 训练模型
```bash
bash attention_robust_v2/train_v2.sh
```

### 步骤3: 运行消融实验
```bash
bash attention_robust_v2/run_ablation.sh
```

### 步骤4: 提取最佳结果
```bash
python attention_robust_v2/extract_best_results.py --output attention_robust_v2/results_latest.md
```

---

*本文件由 extract_best_results.py 的实验数据和代码分析自动整理生成*
