import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# KL 有效区域 Ω：mask 值 < 此阈值表示该图文 token 对未被屏蔽，可参与矛盾值（CV）计算
KL_VALID_THRESHOLD = 0.5


class LearnableMaskGenerator(nn.Module):
    """为每个 (image_token, text_token) 对学习是否屏蔽该对的二值 mask。"""

    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.base_temp = 10.0

        # 根据局部+全局图文特征预测逐对相似度阈值
        self.threshold_net = nn.Sequential(
            nn.Linear(4*d_model, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

        self.register_buffer('temperature', torch.tensor(self.base_temp))
        self.register_buffer('current_epoch', torch.tensor(0))

    def update_temperature(self, epoch, total_epochs):
        """训练过程中逐步降低 Gumbel 温度，使 mask 从软采样趋于离散。"""
        progress = min(epoch / total_epochs, 1.0)

        new_temp = self.base_temp * (1.0 - 0.5 * progress)
        new_temp = max(new_temp, 1.0)

        self.temperature.fill_(new_temp)
        self.current_epoch.fill_(epoch)

    def forward(self, q, k, q_global, k_global, stochastic=None):
        B, L, D = q.shape
        S = k.size(1)

        q_global = q_global.unsqueeze(1).expand(-1, L, -1)
        k_global = k_global.unsqueeze(1).expand(-1, S, -1)

        q_exp = q.unsqueeze(2)           # [B, L, 1, D]
        k_exp = k.unsqueeze(1)           # [B, 1, S, D]
        qg_exp = q_global.unsqueeze(2)   # [B, L, 1, D]
        kg_exp = k_global.unsqueeze(1)   # [B, 1, S, D]

        threshold_input = torch.cat([
            q_exp.expand(-1, -1, S, -1),
            k_exp.expand(-1, L, -1, -1),
            qg_exp.expand(-1, -1, S, -1),
            kg_exp.expand(-1, L, -1, -1)
        ], dim=-1)

        thresholds = self.threshold_net(threshold_input).squeeze(-1)  # [B, L, S]

        q = q.view(B, L, D)
        k = k.view(B, S, D)
        sim_matrix = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)  # [B, L, S]

        # sim > threshold → 高相似（保留）；sim <= threshold → 低相似（屏蔽，关注矛盾区域）
        logits = (sim_matrix - thresholds) / self.temperature  # [B, L, S]

        binary_logits = torch.stack([-logits, logits], dim=-1)  # [B, L, S, 2]
        if stochastic is None:
            stochastic = self.training

        if stochastic:
            # 训练：Gumbel-Softmax 硬采样，保留梯度
            mask = F.gumbel_softmax(
                binary_logits,
                tau=1.0,
                hard=True,
                dim=-1
            )[..., 1].unsqueeze(1)  # [B, 1, L, S]
        else:
            # 验证/推理：argmax 确定性 mask，避免随机性影响评估结果
            mask = (binary_logits.argmax(dim=-1) == 1).to(logits.dtype).unsqueeze(1)

        return mask


class CrossModalAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2*d_model)

        self.img_masker = LearnableMaskGenerator(d_model, num_heads)
        self.txt_masker = LearnableMaskGenerator(d_model, num_heads)

    def forward(self, img_seq, txt_seq, isTrain, img_mask=None, txt_mask=None):
        img_global = img_seq.mean(dim=1)
        txt_global = txt_seq.mean(dim=1)

        # 仅训练阶段使用随机 mask；验证/测试使用固定 argmax mask
        stochastic_mask = self.training and isTrain
        if img_mask is None:
            img_mask = self.img_masker(
                img_seq, txt_seq, img_global, txt_global, stochastic=stochastic_mask
            )
        if txt_mask is None:
            txt_mask = self.txt_masker(
                txt_seq, img_seq, txt_global, img_global, stochastic=stochastic_mask
            )

        img_out = self.masked_attention(
            self.q_proj(img_seq),
            self.kv_proj(txt_seq),
            img_mask
        )
        txt_out = self.masked_attention(
            self.q_proj(txt_seq),
            self.kv_proj(img_seq),
            txt_mask
        )

        total_elements = 197 * 197
        non_zero_counts = txt_mask.sum(dim=(-1, -2))
        txt_sparsity = non_zero_counts.squeeze(1) / total_elements
        non_zero_counts = img_mask.sum(dim=(-1, -2))
        img_sparsity = non_zero_counts.squeeze(1) / total_elements

        img_pool = img_out.mean(dim=1)
        text_pool = txt_out.mean(dim=1)

        fine_grained_consistency = torch.cat([img_pool, text_pool], dim=1)

        # CV 在原始投影特征上计算，mask 仅决定哪些 (i,j) 对进入 KL
        kl_loss = self.bidirectional_kl(img_seq, txt_seq, img_mask, txt_mask)

        return kl_loss, fine_grained_consistency, img_sparsity, txt_sparsity, img_out, txt_out

    def masked_attention(self, q, kv, mask):
        B, L, D = q.shape

        q_heads = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        k_heads = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_heads = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # mask==1 的位置在注意力中被屏蔽（设为 -inf）
        mask_expanded = mask.expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(mask_expanded >= KL_VALID_THRESHOLD, float('-inf'))

        # 若某 query 行全部被屏蔽，softmax 会得到 nan；将该行分数置零回退为均匀注意力
        inf_mask = torch.isinf(scores) & (scores < 0)
        all_inf_rows = inf_mask.all(dim=-1, keepdim=True)
        scores = scores.masked_fill(all_inf_rows, 0.0)

        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, v_heads)
        return output.permute(0, 2, 1, 3).reshape(B, L, D)

    def bidirectional_kl(self, img_feats, txt_feats, img_mask, txt_mask):
        """在 Ω 区域（双向均未屏蔽的 token 对）上计算双向 KL，得到样本级矛盾值 CV。"""
        B, _, L, S = img_mask.shape

        # Ω: image→text 与 text→image 两个方向都未屏蔽的位置
        img_valid = (img_mask.squeeze(1) < KL_VALID_THRESHOLD)
        txt_valid = (txt_mask.squeeze(1) < KL_VALID_THRESHOLD).transpose(1, 2)
        joint_mask = img_valid & txt_valid

        log_p_img = F.log_softmax(img_feats + 1e-8, dim=-1)
        p_text = F.softmax(txt_feats + 1e-8, dim=-1)

        log_p_img_exp = log_p_img.unsqueeze(2)
        p_text_exp = p_text.unsqueeze(1)

        kl_img_text = F.kl_div(
            log_p_img_exp,
            p_text_exp.detach(),
            reduction='none'
        ).sum(-1)

        log_p_text_exp = F.log_softmax(txt_feats + 1e-8, dim=-1).unsqueeze(1)
        p_img_exp = F.softmax(img_feats + 1e-8, dim=-1).unsqueeze(2)
        kl_text_img = F.kl_div(
            log_p_text_exp,
            p_img_exp.detach(),
            reduction='none'
        ).sum(-1).transpose(1, 2)

        kl_total = (kl_img_text + kl_text_img) * 0.5

        masked_kl = kl_total * joint_mask.float()
        valid_counts = joint_mask.sum(dim=(1, 2)) + 1e-8
        kl_loss = masked_kl.sum(dim=(1, 2)) / valid_counts

        return kl_loss
