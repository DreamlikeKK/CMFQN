import torch
import torch.nn as nn
from models.new_co import CrossModalAttention
from torch.nn.functional import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=True)
        
    def forward(self, x):
        return self.fc(x)
    
class EnhancedResidualGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.diff_scale = nn.Parameter(torch.tensor(1.0))  

        self.img_gate = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim),
            nn.Sigmoid() 
        )
        self.text_gate = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, img_feat, text_feat):
        """
        input: 
            img_feat, text_feat: [B, D] 
        output: 
            inconsistency: [B, D] 
        """
        gate_img = self.img_gate(img_feat - text_feat)  
        gate_text = self.text_gate(text_feat - img_feat) 
        
        gated_img = gate_img * img_feat  
        gated_text = gate_text * text_feat  
        
        coarse_grained_inconsistency = torch.cat([gated_text, gated_img], dim=1)
        
        return coarse_grained_inconsistency, gated_img, gated_text   

class Net(nn.Module):
    def __init__(self, 
                 unified_dim=768):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.5))  
        self.theta.data.clamp_(min=0.1, max=2.0)      

        self.img_proj = nn.Sequential(
            nn.Linear(unified_dim, 512),
            nn.GELU()
        )
        self.text_proj = nn.Sequential(
            nn.Linear(unified_dim, 512),
            nn.GELU()
        )
        
        self.co_attention = CrossModalAttention(
            512
        )

        self.coarse_grained_extractor = EnhancedResidualGate(512)

        self.classifier = Classifier(2048)
        self.classifier1 = Classifier(1024)

        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, mae_features, text_outputs, clip_image_features, clip_text_features, idx, labels, isTrain):
        proj_image = self.img_proj(mae_features)
        proj_text = self.text_proj(text_outputs)

        contradiction, fine_grained_consistency, img2text_mask, text2img_mask, co_image_features, co_text_features = self.co_attention(proj_image, proj_text, isTrain)
        
        # w/o DAFQ
        # img_pool = proj_image.mean(dim=1)
        # text_pool = proj_text.mean(dim=1)
        # fine_grained_consistency = torch.cat([img_pool, text_pool], dim=1)
        # device = torch.device("cuda:0")
        # img2text_input = torch.zeros(1, device=device, requires_grad=True)
        # img2text_mask = img2text_input.mean()  
        # text2img_input = torch.zeros(1, device=device, requires_grad=True)
        # text2img_mask = text2img_input.mean()  
        # cont_input = torch.zeros(text_outputs.shape[0], device=device, requires_grad=True)
        # contradiction = cont_input / 1.0  

        clip_image_features = clip_image_features.to(dtype=torch.float32)  
        clip_text_features = clip_text_features.to(dtype=torch.float32)

        coarse_grained_inconsistency, clip_image_features1, clip_text_features1 = self.coarse_grained_extractor(clip_image_features, clip_text_features)

        # w/o GAFA
        # coarse_grained_inconsistency = torch.cat([clip_image_features, clip_text_features], dim=1)

        cos_sim = cosine_similarity(clip_image_features, clip_text_features)
        cos_sim_adjusted = (cos_sim + 1) / 2
        
        final_feats = torch.cat([fine_grained_consistency * (1 - cos_sim_adjusted.unsqueeze(-1)), coarse_grained_inconsistency * cos_sim_adjusted.unsqueeze(-1)], dim=1)

        # w/o DCF
        # final_feats = torch.cat([fine_grained_consistency, coarse_grained_inconsistency], dim=1)

        prediction = self.classifier(final_feats)
        inco_prediction = self.classifier1(coarse_grained_inconsistency)
        co_prediction = self.classifier1(fine_grained_consistency)

        return prediction, img2text_mask, text2img_mask, inco_prediction, co_prediction, cos_sim_adjusted, contradiction, 1, final_feats
        