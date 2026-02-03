"""
变分编码器模块 - 基于P-RMF的概率化特征表示
核心思想：将确定性特征编码转变为高斯分布参数估计 (μ, σ)

Reference: Proxy-Driven Robust Multimodal Sentiment Analysis with Incomplete Data (ACL 2025)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalMLPEncoder(nn.Module):
    """
    变分MLP编码器 - 替代原有的MLPEncoder

    输入: 原始特征 x [B, in_dim]
    输出:
        - z: 采样的潜在变量 [B, hidden_dim] (用于后续处理)
        - mu: 均值 [B, hidden_dim] (稳定语义信息)
        - logvar: 对数方差 [B, hidden_dim] (不确定性度量)
        - std: 标准差 [B, hidden_dim]

    核心改进：
    - 模态完整时: σ ≈ 小值，表示高确信度
    - 模态缺失/噪声时: σ → 大值，表示低确信度
    - V4改进: GELU激活 + LayerNorm + 残差连接
    """
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

        # 分支1: 均值预测 μ
        self.mu_layer = nn.Linear(hidden_size, hidden_size)

        # 分支2: 对数方差预测 log(σ²)
        self.logvar_layer = nn.Linear(hidden_size, hidden_size)

        # 初始化logvar层使初始方差接近1 (log(1) = 0)
        nn.init.zeros_(self.logvar_layer.weight)
        nn.init.zeros_(self.logvar_layer.bias)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧: z = μ + ε × σ
        其中 ε ~ N(0, I)

        这使得采样操作可微分，允许梯度反向传播
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # 推理时直接使用均值，保证输出稳定性
            return mu

    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, in_size]
        Returns:
            z: 采样的潜在变量 [batch_size, hidden_size]
            mu: 均值 [batch_size, hidden_size]
            logvar: 对数方差 [batch_size, hidden_size]
            std: 标准差 [batch_size, hidden_size]
        """
        x = self.drop(x)

        # 第一层 + 残差准备
        residual = self.proj(x)
        h = F.gelu(self.ln1(self.fc1(x)))

        # 第二层 + 残差连接
        h = F.gelu(self.ln2(self.fc2(h))) + residual

        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        # 数值稳定性：限制logvar范围，防止exp爆炸或下溢
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, std


class VariationalLSTMEncoder(nn.Module):
    """
    变分LSTM编码器 - 用于序列特征的变分编码
    """
    def __init__(self, in_size, hidden_size, dropout, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        
        if num_layers == 1:
            rnn_dropout = 0.0
        else:
            rnn_dropout = dropout
            
        self.rnn = nn.LSTM(
            in_size, hidden_size, 
            num_layers=num_layers, 
            dropout=rnn_dropout, 
            bidirectional=bidirectional, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
        # 共享层
        self.shared = nn.Linear(hidden_size, hidden_size)
        
        # 均值和方差分支
        self.mu_layer = nn.Linear(hidden_size, hidden_size)
        self.logvar_layer = nn.Linear(hidden_size, hidden_size)
        
        # 初始化
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
        """
        Args:
            x: 序列特征 [batch_size, seq_len, in_size]
        """
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze(0))
        h = F.relu(self.shared(h))
        
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar, std


class ModalityDecoder(nn.Module):
    """
    模态解码器 - 从潜在变量重建原始特征
    
    作用: 
    - 强制编码器即使在输入残缺时也要保持语义完整性
    - 通过重建损失监督，使潜在表示包含足够的信息
    """
    def __init__(self, hidden_size, out_size, dropout):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, z):
        """
        Args:
            z: 潜在变量 [batch_size, hidden_size]
        Returns:
            reconstructed: 重建的特征 [batch_size, out_size]
        """
        return self.decoder(z)


class UncertaintyWeightedFusion(nn.Module):
    """
    基于不确定性的动态加权融合 - 生成代理模态
    
    核心公式: w_m = softmax(1/σ_m)
    物理意义: 不确定性(方差)越大的模态，融合权重越低
    
    这是P-RMF的核心创新之一
    """
    def __init__(self, hidden_dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        
    def forward(self, mu_list, std_list):
        """
        Args:
            mu_list: [mu_audio, mu_text, mu_video], 每个形状 [B, H]
            std_list: [std_audio, std_text, std_video], 每个形状 [B, H]
        Returns:
            proxy: 代理模态 [B, H]
            weights: 各模态权重 [B, 3]
        """
        # 计算每个模态的平均不确定性
        uncertainties = []
        for std in std_list:
            # 每个模态的平均标准差作为不确定性度量
            uncertainty = std.mean(dim=-1, keepdim=True)  # [B, 1]
            uncertainties.append(uncertainty)
        uncertainties = torch.cat(uncertainties, dim=1)  # [B, 3]
        
        # 反向方差加权: w = softmax(1/σ)
        # 不确定性越低(σ越小)，权重越高
        inv_uncertainties = 1.0 / (uncertainties + 1e-6)
        weights = F.softmax(inv_uncertainties / self.temperature, dim=1)  # [B, 3]
        
        # 加权融合各模态的均值生成代理模态
        mu_stack = torch.stack(mu_list, dim=1)  # [B, 3, H]
        weights_exp = weights.unsqueeze(-1)  # [B, 3, 1]
        proxy = (mu_stack * weights_exp).sum(dim=1)  # [B, H]
        
        return proxy, weights


class GatedUncertaintyFusion(nn.Module):
    """
    门控增强的不确定性加权融合

    改进:
    - 不确定性权重: w_unc = softmax(1/σ)
    - 门控权重: w_gate = softmax(MLP(concat(mu_a, mu_t, mu_v)))
    - 最终权重: w = α * w_unc + (1-α) * w_gate

    这允许模型学习更灵活的融合策略，而不仅仅依赖不确定性
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


class ProxyCrossModalAttention(nn.Module):
    """
    代理模态引导的跨模态注意力
    
    使用proxy作为稳定的Query，对各原始模态做加权attention
    这样即使某个模态缺失，proxy仍能从其他模态获取信息
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 三个模态的Cross Attention
        self.cross_attn_audio = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_video = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, proxy, mu_audio, mu_text, mu_video, weights):
        """
        Args:
            proxy: 代理模态 [B, H]
            mu_audio, mu_text, mu_video: 各模态均值 [B, H]
            weights: 各模态不确定性权重 [B, 3]
        Returns:
            fused: 融合后的特征 [B, H]
        """
        # 扩展维度用于attention: [B, 1, H]
        proxy_exp = proxy.unsqueeze(1)
        audio_exp = mu_audio.unsqueeze(1)
        text_exp = mu_text.unsqueeze(1)
        video_exp = mu_video.unsqueeze(1)
        
        # 分别做cross attention: proxy attend to each modality
        attn_a, _ = self.cross_attn_audio(proxy_exp, audio_exp, audio_exp)
        attn_t, _ = self.cross_attn_text(proxy_exp, text_exp, text_exp)
        attn_v, _ = self.cross_attn_video(proxy_exp, video_exp, video_exp)
        
        # Squeeze回 [B, H]
        attn_a = attn_a.squeeze(1)
        attn_t = attn_t.squeeze(1)
        attn_v = attn_v.squeeze(1)
        
        # 使用不确定性权重加权融合attention结果
        w_a = weights[:, 0:1]  # [B, 1]
        w_t = weights[:, 1:2]
        w_v = weights[:, 2:3]
        
        weighted_attn = w_a * attn_a + w_t * attn_t + w_v * attn_v
        
        # 残差连接 + LayerNorm + FFN
        fused = self.norm(proxy + weighted_attn)
        fused = fused + self.ffn(fused)
        
        return fused


class VAELossComputer:
    """
    VAE损失计算器
    
    总损失 = α * L_KL + β * L_recon + γ * L_cross_KL
    
    各项损失的作用:
    - L_KL: 约束潜在空间接近标准正态分布，起正则化作用
    - L_recon: 强制编码器保持语义完整性
    - L_cross_KL: 鼓励各模态学习相似的潜在空间，便于跨模态补充
    """
    def __init__(self, kl_weight=0.01, recon_weight=0.1, cross_kl_weight=0.01):
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.cross_kl_weight = cross_kl_weight
        self.mse = nn.MSELoss()
    
    def kl_divergence_to_standard_normal(self, mu, logvar):
        """
        计算KL(q(z|x) || N(0,I))
        = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        
        作用: 正则化，防止潜在空间过拟合
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
    
    def reconstruction_loss(self, original, reconstructed):
        """
        重建损失 = MSE(x, x_recon)
        
        作用: 强制编码器保持语义信息
        """
        return self.mse(reconstructed, original)
    
    def cross_modal_kl(self, mu_list, logvar_list):
        """
        跨模态KL散度
        KL(q_a || q_t) + KL(q_a || q_v) + KL(q_t || q_v)
        
        作用: 鼓励各模态学习相似的潜在空间，便于跨模态补充
        """
        def kl_gaussian(mu1, lv1, mu2, lv2):
            """计算两个高斯分布之间的KL散度"""
            var1 = torch.exp(lv1)
            var2 = torch.exp(lv2) + 1e-6
            kl = 0.5 * (lv2 - lv1 + var1/var2 + (mu1-mu2).pow(2)/var2 - 1)
            return kl.mean()
        
        mu_a, mu_t, mu_v = mu_list
        lv_a, lv_t, lv_v = logvar_list
        
        return (kl_gaussian(mu_a, lv_a, mu_t, lv_t) +
                kl_gaussian(mu_a, lv_a, mu_v, lv_v) +
                kl_gaussian(mu_t, lv_t, mu_v, lv_v)) / 3
    
    def compute(self, mu_list, logvar_list, originals, reconstructions):
        """
        计算总的辅助损失
        
        Args:
            mu_list: [mu_a, mu_t, mu_v]
            logvar_list: [logvar_a, logvar_t, logvar_v]
            originals: [audio, text, video] 原始输入
            reconstructions: [recon_a, recon_t, recon_v] 重建结果
        """
        # 1. KL散度损失 (每个模态对标准正态)
        kl_loss = sum(
            self.kl_divergence_to_standard_normal(mu, lv) 
            for mu, lv in zip(mu_list, logvar_list)
        ) / 3
        
        # 2. 重建损失
        recon_loss = sum(
            self.reconstruction_loss(orig, recon) 
            for orig, recon in zip(originals, reconstructions)
        ) / 3
        
        # 3. 跨模态KL损失
        cross_kl_loss = self.cross_modal_kl(mu_list, logvar_list)
        
        total = (self.kl_weight * kl_loss +
                 self.recon_weight * recon_loss +
                 self.cross_kl_weight * cross_kl_loss)

        return total


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
