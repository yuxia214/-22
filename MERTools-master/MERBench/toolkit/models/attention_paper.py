"""
Description: paper-style attention fusion (linear scoring + softmax over modalities)
"""
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder, LSTMEncoder


class AttentionPaper(nn.Module):
    def __init__(self, args):
        super(AttentionPaper, self).__init__()

        text_dim = args.text_dim
        audio_dim = args.audio_dim
        video_dim = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip

        if args.feat_type in ["utt"]:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder = MLPEncoder(text_dim, hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)
        elif args.feat_type in ["frm_align", "frm_unalign"]:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder = LSTMEncoder(text_dim, hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)

        # Paper-style modality scoring: alpha = softmax(W_alpha * concat(h_a, h_l, h_v) + b_alpha)
        self.fc_att = nn.Linear(hidden_dim * 3, 3)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio_hidden = self.audio_encoder(batch["audios"])  # [B, H]
        text_hidden = self.text_encoder(batch["texts"])     # [B, H]
        video_hidden = self.video_encoder(batch["videos"])  # [B, H]

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1)  # [B, 3H]
        attention_logits = self.fc_att(multi_hidden1)                                  # [B, 3]
        attention = torch.softmax(attention_logits, dim=1).unsqueeze(2)                # [B, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2)  # [B, H, 3]
        fused_feat = torch.matmul(multi_hidden2, attention)                             # [B, H, 1]

        features = fused_feat.squeeze(axis=2)  # [B, H]
        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        interloss = torch.tensor(0).cuda()

        return features, emos_out, vals_out, interloss
