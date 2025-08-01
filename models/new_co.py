import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnableMaskGenerator(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.base_temp = 10.0  
        
        self.threshold_net = nn.Sequential(
            nn.Linear(4*d_model, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )

        self.register_buffer('temperature', torch.tensor(self.base_temp))
        self.register_buffer('current_epoch', torch.tensor(0))

    def update_temperature(self, epoch, total_epochs):
        progress = min(epoch / total_epochs, 1.0)
        
        new_temp = self.base_temp * (1.0 - 0.5 * progress)
        new_temp = max(new_temp, 1.0)  
        
        self.temperature.fill_(new_temp)
        self.current_epoch.fill_(epoch)

    def forward(self, q, k, q_global, k_global):
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

        q = q.view(B, L, D)  # [B, L, D]
        k = k.view(B, S, D)  # [B, S, D]
        sim_matrix = torch.matmul(q, k.transpose(1, 2)) / (D ** 0.5)  # [B, L, S]

        logits = (sim_matrix - thresholds) / self.temperature  # [B, L, S]
        
        binary_logits = torch.stack([-logits, logits], dim=-1)  # [B, L, S, 2]
        mask = F.gumbel_softmax(
            binary_logits,
            tau=1.0,
            hard=True,
            dim=-1
        )[..., 1].unsqueeze(1)  # [B, 1, L, S]

        return mask  # [B, 1, L, S]
    

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
        
        if img_mask is None:
            img_mask = self.img_masker(img_seq, txt_seq, img_global, txt_global)
        if txt_mask is None:
            txt_mask = self.txt_masker(txt_seq, img_seq, txt_global, img_global)

        # w/0 mask，0 mask
        # B, L, D = img_seq.shape
        # S = txt_seq.shape[1]
        # img_mask = torch.zeros(B, 1, L, S, device=img_seq.device)
        # txt_mask = torch.zeros(B, 1, S, L, device=txt_seq.device)

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

        # w/o mask
        # img_sparsity = torch.mean(img_mask.float(), dim=1)
        # txt_sparsity = torch.mean(txt_mask.float(), dim=1)

        total_elements = 197 * 197  
        non_zero_counts = txt_mask.sum(dim=(-1, -2))  #  [16, 1]
        txt_sparsity = non_zero_counts.squeeze(1) / total_elements  #  [16]
        non_zero_counts = img_mask.sum(dim=(-1, -2))  #  [16, 1]
        img_sparsity = non_zero_counts.squeeze(1) / total_elements  #  [16]

        img_pool = img_out.mean(dim=1) 
        text_pool = txt_out.mean(dim=1)

        fine_grained_consistency = torch.cat([img_pool, text_pool], dim=1)

        kl_loss = self.bidirectional_kl(img_out, txt_out, img_mask, txt_mask)
        
        return kl_loss, fine_grained_consistency, img_sparsity, txt_sparsity, img_out, txt_out

    def masked_attention(self, q, kv, mask):
        B, L, D = q.shape
        
        q_heads = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, L, d]
        k, v = kv.chunk(2, dim=-1)
        k_heads = k.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, S, d]
        v_heads = v.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, S, d]

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, L, S]
        
        mask_expANDED = mask.expand(-1, self.num_heads, -1, -1)  # [B, H, L, S]
        scores = scores.masked_fill(mask_expANDED >= 0.5, float('-inf'))
        
        inf_mask = torch.isinf(scores) & (scores < 0)  
        all_inf_rows = inf_mask.all(dim=-1, keepdim=True)  # [B, H, L, 1]
        scores = scores.masked_fill(all_inf_rows, 0.0)     
        
        attn = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attn, v_heads)          # [B, H, L, d]
        return output.permute(0, 2, 1, 3).reshape(B, L, D)
    
    def bidirectional_kl(self, img_out, txt_out, img_mask, txt_mask):
        B, _, L, S = img_mask.shape  

        img_mask_joint = (img_mask.sum(dim=1) <= 1)    # [B, L, S]
        txt_mask_joint = (txt_mask.sum(dim=1) <= 1)    # [B, S, L]
        txt_mask_joint = txt_mask_joint.transpose(1, 2)    # [B, L, S]
        joint_mask = img_mask_joint & txt_mask_joint       # [B, L, S]

        log_p_img = F.log_softmax(img_out + 1e-8, dim=-1)  # [B, L, D]
        p_text = F.softmax(txt_out + 1e-8, dim=-1)         # [B, S, D]

        log_p_img_exp = log_p_img.unsqueeze(2)      # [B, L, 1, D]
        p_text_exp = p_text.unsqueeze(1)           # [B, 1, S, D]

        # KL(P_img || P_text)
        kl_img_text = F.kl_div(
            log_p_img_exp, 
            p_text_exp.detach(),
            reduction='none'
        ).sum(-1)  # [B, L, S]

        # KL(P_text || P_img)
        log_p_text_exp = torch.log_softmax(txt_out, dim=-1).unsqueeze(1)  # [B, 1, S, D]
        p_img_exp = F.softmax(img_out, dim=-1).unsqueeze(2)               # [B, L, 1, D]
        kl_text_img = F.kl_div(
            log_p_text_exp,
            p_img_exp.detach(),
            reduction='none'
        ).sum(-1).transpose(1, 2)  # [B, L, S]

        kl_total = (kl_img_text + kl_text_img) * 0.5  # [B, L, S]

        masked_kl = kl_total * joint_mask.float()

        valid_counts = joint_mask.sum(dim=(1,2)) + 1e-8
        kl_loss = masked_kl.sum(dim=(1,2)) / valid_counts  # [B]

        return kl_loss