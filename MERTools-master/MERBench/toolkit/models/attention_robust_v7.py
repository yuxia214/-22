'''
AttentionRobustV7 - 基于P-RMF V6的面向Challenge指标优化版本

目标优化:
- 优先提升 MER-MULTI / MER-NOISE 的 Combined (F1 - 0.25*MSE)
- 保持 MER-SEMI 的离散情感性能

核心改进:
1. 继承V6稳态改造: mu路径模态dropout + 动态KL warmup
2. 新增情感-价度先验融合: valence = blend(raw_valence, emotion_prior_valence)
3. 新增一致性正则: 约束分类结果与回归结果语义一致
4. 新增特征噪声增强: 训练阶段模拟MER-NOISE扰动
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.encoder import MLPEncoder, LSTMEncoder
from .modules.variational_encoder import (
    VariationalMLPEncoder,
    VariationalLSTMEncoder,
    ModalityDecoder,
    UncertaintyWeightedFusion,
    ProxyCrossModalAttention,
    VAELossComputer,
)


class LinearKLScheduler:
    """线性KL warmup调度器"""

    def __init__(self, init_weight=0.0, final_weight=0.01, warmup_epochs=20):
        self.init_weight = init_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs

    def get_weight(self, epoch):
        if self.warmup_epochs <= 0:
            return self.final_weight
        if epoch < self.warmup_epochs:
            return self.init_weight + (self.final_weight - self.init_weight) * (epoch / self.warmup_epochs)
        return self.final_weight


class AttentionRobustV7(nn.Module):
    """
    AttentionRobustV7 - 概率化多模态情感识别模型 (V7)

    架构:
    1. 变分编码层: 各模态编码为高斯分布参数 (mu, std)
    2. 模态dropout + 不确定性放大: 强化缺失/噪声鲁棒性
    3. 不确定性加权融合 + proxy cross-modal attention
    4. 双头输出:
       - emotion 分类头
       - valence 回归头 + emotion先验融合
    5. 辅助损失:
       - VAE: KL + recon + cross-KL
       - valence-consistency: valence 与 emotion先验一致
    """

    def __init__(self, args):
        super(AttentionRobustV7, self).__init__()

        # ==================== 基础参数 ====================
        text_dim = args.text_dim
        audio_dim = args.audio_dim
        video_dim = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.hidden_dim = hidden_dim
        self.grad_clip = args.grad_clip

        # ==================== VAE参数 ====================
        self.use_vae = getattr(args, 'use_vae', True)
        self.kl_weight = getattr(args, 'kl_weight', 0.01)
        self.recon_weight = getattr(args, 'recon_weight', 0.1)
        self.cross_kl_weight = getattr(args, 'cross_kl_weight', 0.01)
        self.use_dynamic_kl = getattr(args, 'use_dynamic_kl', True)
        self.kl_warmup_epochs = getattr(args, 'kl_warmup_epochs', 20)

        # ==================== 代理模态参数 ====================
        self.use_proxy_attention = getattr(args, 'use_proxy_attention', True)
        self.fusion_temperature = getattr(args, 'fusion_temperature', 1.0)
        self.num_attention_heads = getattr(args, 'num_attention_heads', 4)

        # ==================== 模态dropout参数 ====================
        self.modality_dropout = getattr(args, 'modality_dropout', 0.2)
        self.use_modality_dropout = getattr(args, 'use_modality_dropout', True)
        self.warmup_epochs = getattr(args, 'modality_dropout_warmup', 0)
        self.current_epoch = 0

        # ==================== 噪声增强参数 (V7新增) ====================
        self.feature_noise_std = getattr(args, 'feature_noise_std', 0.02)
        self.feature_noise_prob = getattr(args, 'feature_noise_prob', 0.3)
        self.feature_noise_warmup = getattr(args, 'feature_noise_warmup', 10)

        # ==================== 编码器 ====================
        if args.feat_type in ['utt']:
            if self.use_vae:
                self.audio_encoder = VariationalMLPEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = VariationalMLPEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = VariationalMLPEncoder(video_dim, hidden_dim, dropout)

                self.audio_decoder = ModalityDecoder(hidden_dim, audio_dim, dropout)
                self.text_decoder = ModalityDecoder(hidden_dim, text_dim, dropout)
                self.video_decoder = ModalityDecoder(hidden_dim, video_dim, dropout)
            else:
                self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = MLPEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        elif args.feat_type in ['frm_align', 'frm_unalign']:
            if self.use_vae:
                self.audio_encoder = VariationalLSTMEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = VariationalLSTMEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = VariationalLSTMEncoder(video_dim, hidden_dim, dropout)

                self.audio_decoder = ModalityDecoder(hidden_dim, audio_dim, dropout)
                self.text_decoder = ModalityDecoder(hidden_dim, text_dim, dropout)
                self.video_decoder = ModalityDecoder(hidden_dim, video_dim, dropout)
            else:
                self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
                self.text_encoder = LSTMEncoder(text_dim, hidden_dim, dropout)
                self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        # ==================== 融合模块 ====================
        if self.use_vae:
            self.uncertainty_fusion = UncertaintyWeightedFusion(
                hidden_dim, temperature=self.fusion_temperature
            )

            if self.use_proxy_attention:
                self.proxy_attention = ProxyCrossModalAttention(
                    hidden_dim,
                    num_heads=self.num_attention_heads,
                    dropout=dropout,
                )

            self.loss_computer = VAELossComputer(
                kl_weight=self.kl_weight,
                recon_weight=self.recon_weight,
                cross_kl_weight=self.cross_kl_weight,
            )
            if self.use_dynamic_kl:
                self.kl_scheduler = LinearKLScheduler(
                    init_weight=0.0,
                    final_weight=self.kl_weight,
                    warmup_epochs=self.kl_warmup_epochs,
                )
        else:
            self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)
            self.fc_att = nn.Linear(hidden_dim, 3)

        # ==================== 输出层 ====================
        self.feat_dropout = nn.Dropout(p=dropout)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

        # ==================== Emotion-Valence一致性模块 (V7新增) ====================
        self.use_valence_prior = getattr(args, 'use_valence_prior', True)
        self.valence_consistency_weight = getattr(args, 'valence_consistency_weight', 0.08)
        self.valence_center_reg_weight = getattr(args, 'valence_center_reg_weight', 0.005)

        if output_dim1 == 6 and output_dim2 == 1:
            # [neutral, angry, happy, sad, worried, surprise]
            init_centers = torch.tensor([0.0, -2.0, 2.0, -2.0, -1.0, 0.3], dtype=torch.float32)
        else:
            init_centers = torch.zeros(max(output_dim1, 1), dtype=torch.float32)

        self.register_buffer('emo_center_init', init_centers.clone())
        self.emo_valence_centers = nn.Parameter(init_centers.clone())
        self.valence_prior_gate = nn.Parameter(torch.tensor(0.5))

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def apply_feature_noise(self, x):
        """训练期特征高斯噪声增强，用于提升NOISE鲁棒性"""
        if not self.training:
            return x
        if self.current_epoch < self.feature_noise_warmup:
            return x
        if self.feature_noise_std <= 0:
            return x
        if torch.rand(1).item() >= self.feature_noise_prob:
            return x
        noise = torch.randn_like(x) * self.feature_noise_std
        return x + noise

    def apply_modality_dropout(self, z_audio, z_text, z_video):
        """训练期模态dropout"""
        if not self.training or not self.use_modality_dropout:
            return z_audio, z_text, z_video

        if self.current_epoch < self.warmup_epochs:
            return z_audio, z_text, z_video

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

    def fuse_valence_with_emotion_prior(self, emos_out, raw_vals_out):
        """通过emotion分布先验修正valence预测"""
        if not self.use_valence_prior or emos_out.size(1) != self.emo_valence_centers.numel():
            return raw_vals_out, None

        probs = F.softmax(emos_out, dim=1)
        prior_val = torch.sum(probs * self.emo_valence_centers.unsqueeze(0), dim=1, keepdim=True)
        gate = torch.sigmoid(self.valence_prior_gate)
        vals_out = gate * raw_vals_out + (1.0 - gate) * prior_val
        return vals_out, prior_val

    def forward(self, batch):
        if self.use_vae:
            return self.forward_vae(batch)
        return self.forward_original(batch)

    def forward_vae(self, batch):
        audios = self.apply_feature_noise(batch['audios'])
        texts = self.apply_feature_noise(batch['texts'])
        videos = self.apply_feature_noise(batch['videos'])

        # 1) 变分编码
        z_a, mu_a, logvar_a, std_a = self.audio_encoder(audios)
        z_t, mu_t, logvar_t, std_t = self.text_encoder(texts)
        z_v, mu_v, logvar_v, std_v = self.video_encoder(videos)

        # 2) 模态dropout作用到mu
        mu_a_dropped, mu_t_dropped, mu_v_dropped = self.apply_modality_dropout(mu_a, mu_t, mu_v)

        # 被dropout模态对应的不确定性设大
        std_a_adj = torch.where(mu_a_dropped.abs().sum(dim=1, keepdim=True) == 0, torch.ones_like(std_a) * 1e6, std_a)
        std_t_adj = torch.where(mu_t_dropped.abs().sum(dim=1, keepdim=True) == 0, torch.ones_like(std_t) * 1e6, std_t)
        std_v_adj = torch.where(mu_v_dropped.abs().sum(dim=1, keepdim=True) == 0, torch.ones_like(std_v) * 1e6, std_v)

        # 3) 不确定性融合
        proxy, weights = self.uncertainty_fusion(
            [mu_a_dropped, mu_t_dropped, mu_v_dropped],
            [std_a_adj, std_t_adj, std_v_adj],
        )

        # 4) proxy attention
        if self.use_proxy_attention:
            fused = self.proxy_attention(proxy, mu_a_dropped, mu_t_dropped, mu_v_dropped, weights)
        else:
            fused = proxy

        # 5) 输出头
        features = self.feat_dropout(fused)
        emos_out = self.fc_out_1(features)
        raw_vals_out = self.fc_out_2(features)
        vals_out, prior_val = self.fuse_valence_with_emotion_prior(emos_out, raw_vals_out)

        # 6) 辅助损失
        if self.training:
            recon_a = self.audio_decoder(z_a)
            recon_t = self.text_decoder(z_t)
            recon_v = self.video_decoder(z_v)

            if self.use_dynamic_kl:
                self.loss_computer.kl_weight = self.kl_scheduler.get_weight(self.current_epoch)

            interloss = self.loss_computer.compute(
                mu_list=[mu_a, mu_t, mu_v],
                logvar_list=[logvar_a, logvar_t, logvar_v],
                originals=[audios, texts, videos],
                reconstructions=[recon_a, recon_t, recon_v],
            )

            if prior_val is not None:
                consistency = F.smooth_l1_loss(vals_out.view(-1, 1), prior_val.view(-1, 1))
                center_reg = F.mse_loss(self.emo_valence_centers, self.emo_center_init)
                interloss = interloss + self.valence_consistency_weight * consistency + self.valence_center_reg_weight * center_reg
        else:
            interloss = torch.tensor(0.0, device=audios.device)

        return features, emos_out, vals_out, interloss

    def forward_original(self, batch):
        audio_hidden = self.audio_encoder(batch['audios'])
        text_hidden = self.text_encoder(batch['texts'])
        video_hidden = self.video_encoder(batch['videos'])

        audio_hidden, text_hidden, video_hidden = self.apply_modality_dropout(audio_hidden, text_hidden, video_hidden)

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1)
        attention = self.attention_mlp(multi_hidden1)
        attention = self.fc_att(attention)
        attention = F.softmax(attention, dim=1)
        attention = attention.unsqueeze(2)

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2)
        fused_feat = torch.matmul(multi_hidden2, attention)

        features = fused_feat.squeeze(2)
        features = self.feat_dropout(features)

        emos_out = self.fc_out_1(features)
        raw_vals_out = self.fc_out_2(features)
        vals_out, _ = self.fuse_valence_with_emotion_prior(emos_out, raw_vals_out)
        interloss = torch.tensor(0.0, device=batch['audios'].device)

        return features, emos_out, vals_out, interloss
