# AttentionRobustV4 模型准确度提升 - 详细实现计划

## 当前状态
- 已有文件: `attention_robust_v4.py`, `modules/variational_encoder.py`, `train_v4.sh`
- 已有模块: VariationalMLPEncoder, VariationalLSTMEncoder, ModalityDecoder, UncertaintyWeightedFusion, ProxyCrossModalAttention, VAELossComputer

---

## 实现任务清单

### Task 1: 添加对比学习模块 (ModalityContrastiveLoss)
**文件**: `modules/variational_encoder.py`
**位置**: 在 VAELossComputer 类之后添加

```python
class ModalityContrastiveLoss(nn.Module):
    """
    跨模态对比学习损失 - 借鉴MMIM的CPC模块

    目标:
    - 拉近同一样本不同模态的表示
    - 推远不同样本的表示
    - 使用InfoNCE损失
    """
    def __init__(self, hidden_dim, proj_dim=64, temperature=0.07):
        super().__init__()
        self.temperature = temperature

        # 三个投影头: 将hidden_dim映射到proj_dim
        self.proj_audio = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        self.proj_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        self.proj_video = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def info_nce(self, z1, z2):
        """
        InfoNCE损失计算
        z1, z2: [B, proj_dim]
        """
        # L2归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.size(0)

        # 计算相似度矩阵 [B, B]
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature

        # 正样本是对角线元素
        labels = torch.arange(batch_size, device=z1.device)

        # 双向InfoNCE
        loss = (F.cross_entropy(sim_matrix, labels) +
                F.cross_entropy(sim_matrix.t(), labels)) / 2

        return loss

    def forward(self, mu_a, mu_t, mu_v):
        """
        计算三对模态的对比损失
        mu_a, mu_t, mu_v: [B, hidden_dim]
        """
        # 投影到对比空间
        z_a = self.proj_audio(mu_a)
        z_t = self.proj_text(mu_t)
        z_v = self.proj_video(mu_v)

        # 三对对比损失
        loss_at = self.info_nce(z_a, z_t)
        loss_av = self.info_nce(z_a, z_v)
        loss_tv = self.info_nce(z_t, z_v)

        return (loss_at + loss_av + loss_tv) / 3
```

---

### Task 2: 添加门控融合增强 (GatedUncertaintyFusion)
**文件**: `modules/variational_encoder.py`
**位置**: 在 UncertaintyWeightedFusion 类之后添加

```python
class GatedUncertaintyFusion(nn.Module):
    """
    门控增强的不确定性加权融合

    改进:
    - 不确定性权重: w_unc = softmax(1/σ)
    - 门控权重: w_gate = sigmoid(MLP(concat(mu_a, mu_t, mu_v)))
    - 最终权重: w = α * w_unc + (1-α) * w_gate
    """
    def __init__(self, hidden_dim, temperature=1.0, gate_alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.gate_alpha = gate_alpha

        # 门控网络: 输入三个模态拼接，输出3个门控值
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, mu_list, std_list):
        """
        Args:
            mu_list: [mu_audio, mu_text, mu_video], 每个形状 [B, H]
            std_list: [std_audio, std_text, std_video], 每个形状 [B, H]
        Returns:
            proxy: 代理模态 [B, H]
            weights: 各模态权重 [B, 3]
        """
        # 1. 计算不确定性权重
        uncertainties = []
        for std in std_list:
            uncertainty = std.mean(dim=-1, keepdim=True)
            uncertainties.append(uncertainty)
        uncertainties = torch.cat(uncertainties, dim=1)  # [B, 3]

        inv_uncertainties = 1.0 / (uncertainties + 1e-6)
        w_unc = F.softmax(inv_uncertainties / self.temperature, dim=1)  # [B, 3]

        # 2. 计算门控权重
        concat_mu = torch.cat(mu_list, dim=1)  # [B, 3H]
        gate_logits = self.gate_net(concat_mu)  # [B, 3]
        w_gate = F.softmax(gate_logits, dim=1)  # [B, 3]

        # 3. 融合两种权重
        weights = self.gate_alpha * w_unc + (1 - self.gate_alpha) * w_gate

        # 4. 加权融合
        mu_stack = torch.stack(mu_list, dim=1)  # [B, 3, H]
        weights_exp = weights.unsqueeze(-1)  # [B, 3, 1]
        proxy = (mu_stack * weights_exp).sum(dim=1)  # [B, H]

        return proxy, weights
```

---

### Task 3: 添加Focal Loss
**文件**: `modules/variational_encoder.py` (在文件末尾添加)

```python
class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    γ > 0 时，降低易分类样本的权重，聚焦于难分类样本
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重，可以是tensor或None
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=inputs.device)
            else:
                alpha = self.alpha
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss

    将one-hot标签平滑为: (1-ε)*one_hot + ε/K
    防止模型过于自信，提高泛化能力
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] class indices
        """
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # 创建平滑标签
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLabelSmoothingLoss(nn.Module):
    """
    结合Focal Loss和Label Smoothing的损失函数
    """
    def __init__(self, gamma=2.0, smoothing=0.1, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # 创建平滑标签
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # 计算交叉熵
        ce_loss = (-smooth_targets * log_probs).sum(dim=-1)

        # 计算pt (使用原始targets)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal权重
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=inputs.device)
            else:
                alpha = self.alpha
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

---

### Task 4: 改进变分编码器
**文件**: `modules/variational_encoder.py`
**修改**: VariationalMLPEncoder 类

将 ReLU 改为 GELU，添加 LayerNorm 和残差连接:

```python
class VariationalMLPEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout)

        # 改进: 添加LayerNorm和残差连接
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # 投影层(用于残差连接，当in_size != hidden_size时)
        self.proj = nn.Linear(in_size, hidden_size) if in_size != hidden_size else nn.Identity()

        # 分支: 均值和方差
        self.mu_layer = nn.Linear(hidden_size, hidden_size)
        self.logvar_layer = nn.Linear(hidden_size, hidden_size)

        nn.init.zeros_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        x = self.drop(x)

        # 第一层 + 残差
        residual = self.proj(x)
        h = F.gelu(self.ln1(self.fc1(x)))

        # 第二层 + 残差
        h = F.gelu(self.ln2(self.fc2(h))) + residual

        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, std
```

---

### Task 5: 修改主模型 attention_robust_v4.py
**修改内容**:

1. 导入新模块:
```python
from modules.variational_encoder import (
    ...,
    ModalityContrastiveLoss,
    GatedUncertaintyFusion,
    FocalLabelSmoothingLoss
)
```

2. 在 `__init__` 中添加:
```python
# 对比学习参数
self.use_contrastive = getattr(args, 'use_contrastive', True)
self.contrastive_weight = getattr(args, 'contrastive_weight', 0.1)
self.contrastive_temperature = getattr(args, 'contrastive_temperature', 0.07)

# 门控融合参数
self.use_gated_fusion = getattr(args, 'use_gated_fusion', True)
self.gate_alpha = getattr(args, 'gate_alpha', 0.5)

# 初始化对比学习模块
if self.use_contrastive:
    self.contrastive_loss = ModalityContrastiveLoss(
        hidden_dim,
        temperature=self.contrastive_temperature
    )

# 替换融合模块
if self.use_gated_fusion:
    self.uncertainty_fusion = GatedUncertaintyFusion(
        hidden_dim,
        temperature=self.fusion_temperature,
        gate_alpha=self.gate_alpha
    )
else:
    self.uncertainty_fusion = UncertaintyWeightedFusion(
        hidden_dim,
        temperature=self.fusion_temperature
    )
```

3. 在 `forward_vae` 中添加对比损失:
```python
if self.training:
    # ... 原有损失计算 ...

    # 添加对比损失
    if self.use_contrastive:
        contrastive_loss = self.contrastive_loss(mu_a, mu_t, mu_v)
        interloss = interloss + self.contrastive_weight * contrastive_loss
```

---

### Task 6: 更新训练脚本 train_v4.sh
**添加新参数**:

```bash
# 对比学习参数
USE_CONTRASTIVE="--use_contrastive"
CONTRASTIVE_WEIGHT=0.1
CONTRASTIVE_TEMP=0.07

# 门控融合参数
USE_GATED_FUSION="--use_gated_fusion"
GATE_ALPHA=0.5

# Focal Loss参数
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.1
```

在python命令中添加:
```bash
    ${USE_CONTRASTIVE} \
    --contrastive_weight=${CONTRASTIVE_WEIGHT} \
    --contrastive_temperature=${CONTRASTIVE_TEMP} \
    ${USE_GATED_FUSION} \
    --gate_alpha=${GATE_ALPHA} \
    --focal_gamma=${FOCAL_GAMMA} \
    --label_smoothing=${LABEL_SMOOTHING} \
```

---

## 实现顺序

1. **Step 1**: 修改 `modules/variational_encoder.py`
   - 改进 VariationalMLPEncoder (添加GELU, LayerNorm, 残差)
   - 添加 GatedUncertaintyFusion 类
   - 添加 ModalityContrastiveLoss 类
   - 添加 FocalLoss, LabelSmoothingCrossEntropy, FocalLabelSmoothingLoss 类

2. **Step 2**: 修改 `attention_robust_v4.py`
   - 更新导入
   - 添加新参数
   - 初始化新模块
   - 修改forward_vae添加对比损失

3. **Step 3**: 修改 `train_v4.sh`
   - 添加新超参数配置

---

## 验证检查点

- [x] VariationalMLPEncoder 改进完成
- [x] GatedUncertaintyFusion 添加完成
- [x] ModalityContrastiveLoss 添加完成
- [x] FocalLoss 系列添加完成
- [x] attention_robust_v4.py 集成完成
- [x] train_v4.sh 更新完成
- [x] 代码无语法错误
- [ ] 可以正常运行训练

---

## 文件修改摘要

| 文件 | 操作 | 内容 |
|------|------|------|
| `modules/variational_encoder.py` | 修改+添加 | 改进编码器, 添加4个新类 |
| `attention_robust_v4.py` | 修改 | 集成新模块, 更新损失计算 |
| `train_v4.sh` | 修改 | 添加新超参数 |
