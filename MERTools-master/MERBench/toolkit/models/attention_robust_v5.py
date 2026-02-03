'''
AttentionRobustV5 - 基于Pure VAE的增强版本

核心改进（基于V4消融实验结论）：
1. 保留Pure VAE架构（无对比学习、无门控融合）
2. 新增Mixup数据增强 - 特征级混合提升泛化
3. 动态KL权重调度 - 训练初期低权重，后期逐渐增加
4. 更深的编码器 - 3层MLP + 残差连接
5. 自适应不确定性阈值 - 更好地处理模态缺失
6. 特征重加权 - 基于模态质量的动态加权

Reference:
- V4消融实验：Pure VAE在Test2上达到0.7832（最佳）
- Mixup: Beyond Empirical Risk Minimization (ICLR 2018)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 动态导入
try:
    from toolkit.models.modules.encoder import MLPEncoder, LSTMEncoder
except ImportError:
    from modules.encoder import MLPEncoder, LSTMEncoder


class DeepVariationalEncoder(nn.Module):
    """
    深度变分编码器 - V5改进版

    改进点：
    1. 3层MLP（比V4多一层）
    2. 更强的残差连接
    3. Dropout在每层之间
    """
    def __init__(self, in_size, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size

        # 三层编码器
        self.fc1 = nn.Linear(in_size, hidden_size * 2)
        self.ln1 = nn.LayerNorm(hidden_size * 2)
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.drop2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        # 投影层（用于残差连接）
        self.proj = nn.Linear(in_size, hidden_size)

        # 均值和方差分支
        self.mu_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.logvar_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 初始化logvar层
        nn.init.zeros_(self.logvar_layer[-1].weight)
        nn.init.zeros_(self.logvar_layer[-1].bias)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # 残差准备
        residual = self.proj(x)

        # 三层编码
        h = F.gelu(self.ln1(self.fc1(x)))
        h = self.drop1(h)

        h = F.gelu(self.ln2(self.fc2(h)))
        h = self.drop2(h)

        h = F.gelu(self.ln3(self.fc3(h))) + residual

        # 均值和方差
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar, std


class AdaptiveUncertaintyFusion(nn.Module):
    """
    自适应不确定性融合 - V5改进版

    改进点：
    1. 可学习的不确定性阈值
    2. 软掩码而非硬掩码
    3. 温度参数可学习
    """
    def __init__(self, hidden_dim, init_temperature=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(init_temperature))

        # 可学习的不确定性阈值（用于软掩码）
        self.uncertainty_threshold = nn.Parameter(torch.tensor(1.0))

        # 模态质量评估网络
        self.quality_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, mu_list, std_list):
        """
        Args:
            mu_list: [mu_audio, mu_text, mu_video]
            std_list: [std_audio, std_text, std_video]
        Returns:
            proxy: 代理模态
            weights: 融合权重
            quality_scores: 模态质量分数
        """
        batch_size = mu_list[0].size(0)
        device = mu_list[0].device

        # 计算每个模态的不确定性和质量
        uncertainties = []
        quality_scores = []

        for mu, std in zip(mu_list, std_list):
            # 平均不确定性
            uncertainty = std.mean(dim=-1, keepdim=True)
            uncertainties.append(uncertainty)

            # 模态质量评估
            quality = self.quality_net(mu)
            quality_scores.append(quality)

        uncertainties = torch.cat(uncertainties, dim=1)  # [B, 3]
        quality_scores_tensor = torch.cat(quality_scores, dim=1)  # [B, 3]

        # 软掩码：高不确定性的模态权重降低
        soft_mask = torch.sigmoid(self.uncertainty_threshold - uncertainties)

        # 反向方差加权
        inv_uncertainties = 1.0 / (uncertainties + 1e-6)

        # 结合不确定性权重和质量分数
        combined_weights = inv_uncertainties * soft_mask * quality_scores_tensor
        weights = F.softmax(combined_weights / self.temperature, dim=1)

        # 加权融合
        mu_stack = torch.stack(mu_list, dim=1)  # [B, 3, H]
        weights_exp = weights.unsqueeze(-1)  # [B, 3, 1]
        proxy = (mu_stack * weights_exp).sum(dim=1)  # [B, H]

        return proxy, weights, quality_scores_tensor


class ModalityDecoder(nn.Module):
    """模态解码器"""
    def __init__(self, hidden_size, out_size, dropout):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, z):
        return self.decoder(z)


class DynamicKLScheduler:
    """
    动态KL权重调度器

    策略：训练初期KL权重低（让模型自由学习），后期逐渐增加（正则化）
    """
    def __init__(self, init_weight=0.0, final_weight=0.01, warmup_epochs=20, total_epochs=100):
        self.init_weight = init_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_weight(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增加
            return self.init_weight + (self.final_weight - self.init_weight) * (epoch / self.warmup_epochs)
        else:
            # 后期：保持最终权重
            return self.final_weight


class ProxyCrossModalAttentionV5(nn.Module):
    """
    代理模态跨模态注意力 - V5版本

    改进：添加残差连接和更深的FFN
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 更深的FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, proxy, mu_list, weights):
        """
        Args:
            proxy: 代理模态 [B, H]
            mu_list: [mu_a, mu_t, mu_v]
            weights: 融合权重 [B, 3]
        """
        # 拼接所有模态作为KV
        modalities = torch.stack(mu_list, dim=1)  # [B, 3, H]
        proxy_exp = proxy.unsqueeze(1)  # [B, 1, H]

        # 跨模态注意力
        attn_out, _ = self.cross_attn(proxy_exp, modalities, modalities)
        attn_out = attn_out.squeeze(1)  # [B, H]

        # 残差连接 + LayerNorm
        fused = self.norm1(proxy + attn_out)

        # FFN + 残差
        fused = self.norm2(fused + self.ffn(fused))

        return fused


class AttentionRobustV5(nn.Module):
    """
    AttentionRobustV5 - 基于Pure VAE的增强版本

    核心特点：
    1. 深度变分编码器（3层）
    2. 自适应不确定性融合
    3. Mixup数据增强
    4. 动态KL权重调度
    5. 无对比学习、无门控融合（基于V4消融实验结论）
    """

    def __init__(self, args):
        super(AttentionRobustV5, self).__init__()

        # 基础参数
        text_dim = args.text_dim
        audio_dim = args.audio_dim
        video_dim = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.hidden_dim = hidden_dim
        self.grad_clip = args.grad_clip

        # VAE参数
        self.kl_weight = getattr(args, 'kl_weight', 0.01)
        self.recon_weight = getattr(args, 'recon_weight', 0.1)
        self.cross_kl_weight = getattr(args, 'cross_kl_weight', 0.01)

        # V5新增：动态KL调度
        self.use_dynamic_kl = getattr(args, 'use_dynamic_kl', True)
        self.kl_warmup_epochs = getattr(args, 'kl_warmup_epochs', 20)

        # V5新增：Mixup参数
        self.use_mixup = getattr(args, 'use_mixup', True)
        self.mixup_alpha = getattr(args, 'mixup_alpha', 0.4)

        # 代理模态参数
        self.use_proxy_attention = getattr(args, 'use_proxy_attention', True)
        self.fusion_temperature = getattr(args, 'fusion_temperature', 1.0)
        self.num_attention_heads = getattr(args, 'num_attention_heads', 4)

        # 模态Dropout参数
        self.modality_dropout = getattr(args, 'modality_dropout', 0.15)
        self.use_modality_dropout = getattr(args, 'use_modality_dropout', True)
        self.warmup_epochs = getattr(args, 'modality_dropout_warmup', 20)
        self.current_epoch = 0

        # 深度变分编码器
        self.audio_encoder = DeepVariationalEncoder(audio_dim, hidden_dim, dropout)
        self.text_encoder = DeepVariationalEncoder(text_dim, hidden_dim, dropout)
        self.video_encoder = DeepVariationalEncoder(video_dim, hidden_dim, dropout)

        # 解码器
        self.audio_decoder = ModalityDecoder(hidden_dim, audio_dim, dropout)
        self.text_decoder = ModalityDecoder(hidden_dim, text_dim, dropout)
        self.video_decoder = ModalityDecoder(hidden_dim, video_dim, dropout)

        # 自适应不确定性融合
        self.uncertainty_fusion = AdaptiveUncertaintyFusion(
            hidden_dim,
            init_temperature=self.fusion_temperature
        )

        # 代理模态跨模态注意力
        if self.use_proxy_attention:
            self.proxy_attention = ProxyCrossModalAttentionV5(
                hidden_dim,
                num_heads=self.num_attention_heads,
                dropout=dropout
            )

        # 输出层
        self.feat_dropout = nn.Dropout(p=dropout)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

        # 动态KL调度器
        if self.use_dynamic_kl:
            self.kl_scheduler = DynamicKLScheduler(
                init_weight=0.0,
                final_weight=self.kl_weight,
                warmup_epochs=self.kl_warmup_epochs
            )

    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.current_epoch = epoch

    def mixup_data(self, audios, texts, videos, labels_emo, labels_val):
        """
        Mixup数据增强

        对特征进行线性插值混合，同时混合标签
        """
        if not self.training or not self.use_mixup:
            return audios, texts, videos, labels_emo, labels_val, None

        batch_size = audios.size(0)

        # 从Beta分布采样混合系数
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0

        # 随机打乱索引
        index = torch.randperm(batch_size, device=audios.device)

        # 混合特征
        mixed_audios = lam * audios + (1 - lam) * audios[index]
        mixed_texts = lam * texts + (1 - lam) * texts[index]
        mixed_videos = lam * videos + (1 - lam) * videos[index]

        # 返回混合后的数据和混合信息
        mixup_info = {
            'lam': lam,
            'index': index,
            'labels_emo_b': labels_emo[index] if labels_emo is not None else None,
            'labels_val_b': labels_val[index] if labels_val is not None else None
        }

        return mixed_audios, mixed_texts, mixed_videos, labels_emo, labels_val, mixup_info

    def apply_modality_dropout(self, z_audio, z_text, z_video):
        """模态dropout"""
        if not self.training or not self.use_modality_dropout:
            return z_audio, z_text, z_video

        if self.current_epoch < self.warmup_epochs:
            return z_audio, z_text, z_video

        # 计算有效dropout率
        if self.warmup_epochs > 0:
            progress = min(1.0, (self.current_epoch - self.warmup_epochs) / self.warmup_epochs)
            effective_dropout = self.modality_dropout * progress
        else:
            effective_dropout = self.modality_dropout

        batch_size = z_audio.size(0)
        device = z_audio.device

        masks = torch.ones(batch_size, 3, device=device)

        for i in range(batch_size):
            if torch.rand(1).item() < effective_dropout:
                drop_mode = torch.randint(0, 6, (1,)).item()
                if drop_mode == 0:
                    masks[i, 0] = 0
                elif drop_mode == 1:
                    masks[i, 1] = 0
                elif drop_mode == 2:
                    masks[i, 2] = 0
                elif drop_mode == 3:
                    masks[i, 0] = 0
                    masks[i, 1] = 0
                elif drop_mode == 4:
                    masks[i, 0] = 0
                    masks[i, 2] = 0
                elif drop_mode == 5:
                    masks[i, 1] = 0
                    masks[i, 2] = 0

        z_audio = z_audio * masks[:, 0:1]
        z_text = z_text * masks[:, 1:2]
        z_video = z_video * masks[:, 2:3]

        return z_audio, z_text, z_video

    def compute_vae_loss(self, mu_list, logvar_list, originals, reconstructions):
        """计算VAE损失"""
        # 获取当前KL权重
        if self.use_dynamic_kl:
            current_kl_weight = self.kl_scheduler.get_weight(self.current_epoch)
        else:
            current_kl_weight = self.kl_weight

        # KL散度损失
        kl_loss = 0
        for mu, logvar in zip(mu_list, logvar_list):
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl_loss += kl.mean()
        kl_loss /= 3

        # 重建损失
        recon_loss = 0
        mse = nn.MSELoss()
        for orig, recon in zip(originals, reconstructions):
            recon_loss += mse(recon, orig)
        recon_loss /= 3

        # 跨模态KL损失
        def kl_gaussian(mu1, lv1, mu2, lv2):
            var1 = torch.exp(lv1)
            var2 = torch.exp(lv2) + 1e-6
            kl = 0.5 * (lv2 - lv1 + var1/var2 + (mu1-mu2).pow(2)/var2 - 1)
            return kl.mean()

        mu_a, mu_t, mu_v = mu_list
        lv_a, lv_t, lv_v = logvar_list
        cross_kl = (kl_gaussian(mu_a, lv_a, mu_t, lv_t) +
                    kl_gaussian(mu_a, lv_a, mu_v, lv_v) +
                    kl_gaussian(mu_t, lv_t, mu_v, lv_v)) / 3

        total = (current_kl_weight * kl_loss +
                 self.recon_weight * recon_loss +
                 self.cross_kl_weight * cross_kl)

        return total

    def forward(self, batch):
        """前向传播"""
        audios = batch['audios']
        texts = batch['texts']
        videos = batch['videos']

        # 变分编码
        z_a, mu_a, logvar_a, std_a = self.audio_encoder(audios)
        z_t, mu_t, logvar_t, std_t = self.text_encoder(texts)
        z_v, mu_v, logvar_v, std_v = self.video_encoder(videos)

        # 模态Dropout
        mu_a_dropped, mu_t_dropped, mu_v_dropped = self.apply_modality_dropout(mu_a, mu_t, mu_v)

        # 调整std
        std_a_adj = torch.where(mu_a_dropped.abs().sum(dim=1, keepdim=True) == 0,
                                torch.ones_like(std_a) * 1e6, std_a)
        std_t_adj = torch.where(mu_t_dropped.abs().sum(dim=1, keepdim=True) == 0,
                                torch.ones_like(std_t) * 1e6, std_t)
        std_v_adj = torch.where(mu_v_dropped.abs().sum(dim=1, keepdim=True) == 0,
                                torch.ones_like(std_v) * 1e6, std_v)

        # 自适应不确定性融合
        proxy, weights, quality_scores = self.uncertainty_fusion(
            [mu_a_dropped, mu_t_dropped, mu_v_dropped],
            [std_a_adj, std_t_adj, std_v_adj]
        )

        # 代理模态跨模态注意力
        if self.use_proxy_attention:
            fused = self.proxy_attention(proxy, [mu_a_dropped, mu_t_dropped, mu_v_dropped], weights)
        else:
            fused = proxy

        # 输出
        features = self.feat_dropout(fused)
        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        # 计算辅助损失
        if self.training:
            recon_a = self.audio_decoder(z_a)
            recon_t = self.text_decoder(z_t)
            recon_v = self.video_decoder(z_v)

            interloss = self.compute_vae_loss(
                mu_list=[mu_a, mu_t, mu_v],
                logvar_list=[logvar_a, logvar_t, logvar_v],
                originals=[audios, texts, videos],
                reconstructions=[recon_a, recon_t, recon_v]
            )
        else:
            interloss = torch.tensor(0.0, device=audios.device)

        return features, emos_out, vals_out, interloss
