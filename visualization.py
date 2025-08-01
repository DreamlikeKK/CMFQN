import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import os
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cn_clip.clip as clip
from process_data import models_mae
from matplotlib.colors import LinearSegmentedColormap
from models.Net import Net
import traceback
import torch.nn as nn
from transformers import BertModel, CLIPProcessor
from collections import defaultdict
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPImageProcessor, CLIPProcessor

class GradCAMVisualizer:
    def __init__(self, fnd_model, lang='cn', device='cuda', image_size=224):
        self.fnd_model = fnd_model.to(device).eval()
        self.lang = lang
        self.device = device
        self.image_size = image_size
        
        if lang == 'cn':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese' )
            CN_CLIP_MODEL_NAME = "ViT-B-16"
            CN_CLIP_DOWNLOAD_ROOT = "/home/shunlizhang/zwk/CMFQN/process_data"
            self.clip_model, self.clip_preprocess = clip.load_from_name(
                name=CN_CLIP_MODEL_NAME,
                device=device,
                download_root=CN_CLIP_DOWNLOAD_ROOT
            )
            self.clip_model = self.clip_model.to(device).eval()
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' )
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32" )
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32" ).to(device).eval()
        
        self.mae_model = models_mae.mae_vit_base_patch16(norm_pix_loss=False)
        mae_path = os.path.join(os.getcwd(), 'process_data/mae_pretrain_vit_base.pth')
        checkpoint = torch.load(mae_path, map_location='cpu')
        self.mae_model.load_state_dict(checkpoint['model'], strict=False)
        self.mae_model = self.mae_model.to(device).eval()
        
        if lang == 'cn':
            self.bert_model = BertModel.from_pretrained('bert-base-chinese' ).to(device).eval()
        else:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased' ).to(device).eval()
        
        self.transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2()
        ])
        
        self.features = {}
        self.gradients = {}
        self._register_hooks()
    
    def _register_hooks(self):
        def img_proj_hook(module, input, output):
            self.features['proj_image'] = input[0].detach()
        def text_proj_hook(module, input, output):
            self.features['proj_text'] = input[0].detach()
        
        self.fnd_model.img_proj.register_forward_hook(
            lambda module, input, output: img_proj_hook(module, input, output))
        self.fnd_model.text_proj.register_forward_hook(
            lambda module, input, output: text_proj_hook(module, input, output))
        
        def img_proj_grad_hook(grad):
            self.gradients['img_proj_grad'] = grad
        def text_proj_grad_hook(grad):
            self.gradients['text_proj_grad'] = grad
        
        self.fnd_model.img_proj.register_full_backward_hook(
            lambda module, grad_input, grad_output: img_proj_grad_hook(grad_output[0]))
        self.fnd_model.text_proj.register_full_backward_hook(
            lambda module, grad_input, grad_output: text_proj_grad_hook(grad_output[0]))
    
    def preprocess_text(self, text, max_length=197):
        text = text.strip().lower()
        text = re.sub(r"([,.'!?\"()*#:;~])", '', text)
        text = re.sub(r"-", ' ', text)
        text = re.sub(r"/", ' ', text)
        text = re.sub(r"\s{2,}", ' ', text)
        text = text.rstrip('\n').strip(' ')
        tokens = text.split()[:max_length]
        return ' '.join(tokens)
    
    def preprocess_image(self, image_path):
        pil_image = Image.open(image_path).convert('RGB')
        
        orig_width, orig_height = pil_image.size
        orig_size = (orig_width, orig_height)
        orig_aspect_ratio = orig_width / orig_height
        
        image_np = np.array(pil_image)
        
        transformed = self.transform(image=image_np)
        mae_image = transformed['image'].float().to(self.device)
        
        if self.lang == 'cn':
            clip_image = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        else:
            inputs = self.clip_processor(images=pil_image, return_tensors="pt")
            clip_image = inputs['pixel_values'].to(self.device)
        
        return pil_image, mae_image, clip_image, orig_size, orig_aspect_ratio
    
    def combine_tokens_to_words(self, input_ids, token_scores):
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        word_scores = defaultdict(list)
        current_word = []
        current_word_scores = []
        word_index = 0
        
        for idx, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if token.startswith('##'):
                current_word.append(token[2:])
                current_word_scores.append(token_scores[idx])
            else:
                if current_word:
                    combined_word = ''.join(current_word)
                    word_scores[word_index] = {
                        'word': combined_word,
                        'scores': current_word_scores,
                        'avg_score': np.mean(current_word_scores)
                    }
                    word_index += 1
                
                current_word = [token]
                current_word_scores = [token_scores[idx]]
        
        if current_word:
            combined_word = ''.join(current_word)
            word_scores[word_index] = {
                'word': combined_word,
                'scores': current_word_scores,
                'avg_score': np.mean(current_word_scores)
            }
        
        sorted_words = sorted(word_scores.values(), key=lambda x: x['avg_score'], reverse=True)
        return sorted_words[:3]
    
    def compute_mae_features(self, image):
        with torch.no_grad():
            features = self.mae_model.forward_ying(image.float())
        return features
    
    def compute_bert_features(self, text):
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=197,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state, inputs
    
    def compute_clip_features(self, image, text):
        with torch.no_grad():
            if self.lang == 'cn':
                text_tokens = clip.tokenize([text], context_length=64).to(self.device)
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text_tokens)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                text_inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
                image_features = self.clip_model.get_image_features(image)
                text_features = self.clip_model.get_text_features(**text_inputs)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features
    
    def compute_gradcam(self):
        if 'proj_image' not in self.features or 'img_proj_grad' not in self.gradients:
            return None, None
        
        image_features = self.features['proj_image']
        image_grads = self.gradients['img_proj_grad']
        text_features = self.features['proj_text']
        text_grads = self.gradients['text_proj_grad']
        
        weights = torch.mean(image_grads, dim=(1, 2), keepdim=True)
        image_importance = torch.sum(weights * image_features, dim=-1).squeeze(0)
        
        text_weights = torch.mean(text_grads, dim=-1, keepdim=True)
        text_importance = torch.sum(text_weights * text_features, dim=-1).squeeze(0)
        
        return image_importance.cpu().numpy(), text_importance.cpu().numpy()
    
    def generate_image_heatmap(self, cam, original_image, orig_size):
        if len(cam) == 197:  
            cam = cam[1:]  
        
        grid_size = int(np.sqrt(len(cam)))
        if grid_size * grid_size != len(cam):
            cam = cam[:grid_size*grid_size] if len(cam) > grid_size*grid_size else \
                np.pad(cam, (0, grid_size*grid_size - len(cam)), 'constant')
        
        try:
            cam = cam.reshape(grid_size, grid_size)
        except:
            print(f"The size of CAM cannot be reshaped to {len(cam)} as a square, and one-dimensional averaging is used")
            cam = np.ones((grid_size, grid_size)) * np.mean(cam)
        
        cam = np.maximum(cam, 0)  
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        heatmap = cv2.resize(cam, orig_size, interpolation=cv2.INTER_CUBIC)
        
        colors = [
            (0.0, 0.0, 0.5, 0.0),    
            (0.0, 0.3, 0.8, 0.3),    
            (0.0, 0.6, 0.9, 0.5),     
            (0.0, 0.8, 0.7, 0.7),    
            (0.5, 0.9, 0.5, 0.8),     
            (1.0, 0.95, 0.4, 1.0)    
        ]
        cmap_with_alpha = LinearSegmentedColormap.from_list('jet_alpha', colors)
        
        heatmap_colored = cmap_with_alpha(heatmap)
        
        img_resized = np.array(original_image.resize(orig_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        if img_normalized.shape[-1] == 3:
            img_normalized = np.dstack((img_normalized, np.ones(img_normalized.shape[:2]))) 
        
        alpha = heatmap_colored[..., 3:] 
        rgb = heatmap_colored[..., :3]   
        
        overlay = (1 - alpha) * img_normalized[..., :3] + alpha * rgb
        
        overlay_image = (overlay * 255).astype(np.uint8)
        pil_overlay = Image.fromarray(overlay_image)
        
        return pil_overlay, rgb, cam

    def visualize(self, image_path, text, output_path=None):
        try:
            pil_image, mae_image, clip_image, orig_size, orig_aspect_ratio = self.preprocess_image(image_path)
            
            cleaned_text = self.preprocess_text(text)
            
            mae_features = self.compute_mae_features(mae_image.unsqueeze(0))
            
            bert_features, bert_inputs = self.compute_bert_features(cleaned_text)
            
            clip_image_features, clip_text_features = self.compute_clip_features(
                clip_image, cleaned_text
            )
            
            self.fnd_model.zero_grad()
            
            with torch.enable_grad():
                outputs = self.fnd_model(
                    mae_features,
                    bert_features,
                    clip_image_features,
                    clip_text_features,
                    torch.tensor([0]).to(self.device),
                    None,
                    False
                )
                
                logits = outputs[0]
                labels = torch.zeros(1, dtype=torch.float, device=self.device).unsqueeze(1)
                labels.fill_(0.5)
                
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels)
                
                loss.backward()
                
            image_importance, text_importance = self.compute_gradcam()
            
            if image_importance is None or text_importance is None:
                print("The importance of Grad-CAM cannot be calculated. Use a zero value instead")
                image_importance = np.zeros(197)
                text_importance = np.zeros(197)

            important_words = self.combine_tokens_to_words(
                bert_inputs.input_ids.cpu().numpy(),
                text_importance
            )
            
            overlay_pil, _, _ = self.generate_image_heatmap(
                image_importance, 
                pil_image,
                orig_size=orig_size  
            )
            
            if output_path:
                _, ext = os.path.splitext(output_path)
                ext = ext.lower().replace('.', '')
                
                quality = 95  
                
                if ext in ['jpg', 'jpeg']:
                    overlay_pil.save(output_path, format='JPEG', quality=quality)
                elif ext == 'png':
                    overlay_pil.save(output_path, format='PNG', compress_level=0) 
                elif ext == 'tiff':
                    overlay_pil.save(output_path, format='TIFF', compression=None)  
                else:
                    overlay_pil.save(output_path, format='PNG', compress_level=0)
                
            return True, overlay_pil
            
        except Exception as e:
            print(f"An error occurred during the visualization process: {str(e)}")
            print(traceback.format_exc())
            return False, None

def main(image, text, output, model_path, lang):
    seed = 25
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fnd_model = Net(unified_dim=768)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        if any(key.startswith('module.') for key in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        fnd_model.load_state_dict(state_dict)
    except Exception as e:
        print(traceback.format_exc())
        return
    
    fnd_model = fnd_model.to(device).eval()
    
    visualizer = GradCAMVisualizer(fnd_model, lang=lang, device=device)
    
    success, overlay_image = visualizer.visualize(
        image_path=image,
        text=text,
        output_path=output
    )
    
    if success:
        if output is None and overlay_image is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(overlay_image)
            plt.axis('off')
            plt.title("Grad-CAM Visualization")
            plt.show()
    else:
        print("Visualization failure!")

if __name__ == "__main__":
    arr = [
        {
            'image': '/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/rumor_images/713a4e0bjw1dxi0cvj9b6j.jpg',
            'text': '红海巨蛇： 红海巨大的蛇被埃及科学家终于抓到了。 这条蛇是世界上最长的一条，巨大蛇害死了320游客和125埃及潜水员，八名埃及潜水员合作杀害了红海巨蛇！ 这八位潜水员就是埃及的骄傲',
            'output': 'gradcam_visualization3.png',
            'model_path': '/home/shunlizhang/zwk/CMFQN/output/weibo/data_m0.0_q0.7_e_64_0.9264273101824603.pth',
            'lang': 'cn'
        }, 
        {
            'image': '/home/shunlizhang/zwk/CMFQN/data/data/test/img/31.jpg',
            'text': 'FACT FOCUS: No head trauma or suspicious circumstances in drowning of Obamas’ chef, police say',
            'output': 'gradcam_visualization5.png',
            'model_path': '/home/shunlizhang/zwk/CMFQN/output/ikcest/keshi_q0.8_e_75_0.8483709273182958.pth',
            'lang': 'en'
        }
    ]
        
    wo_arr = [
        {
            'image': '/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/rumor_images/713a4e0bjw1dxi0cvj9b6j.jpg',
            'text': '红海巨蛇： 红海巨大的蛇被埃及科学家终于抓到了。 这条蛇是世界上最长的一条，巨大蛇害死了320游客和125埃及潜水员，八名埃及潜水员合作杀害了红海巨蛇！ 这八位潜水员就是埃及的骄傲',
            'output': 'gradcam_wo_visualization3.png',
            'model_path': '/home/shunlizhang/zwk/CMFQN/output/weibo/womcv_q0.7_e_12_0.9234844025897587.pth',
            'lang': 'cn'
        },
        {
            'image': '/home/shunlizhang/zwk/CMFQN/data/data/test/img/31.jpg',
            'text': 'FACT FOCUS: No head trauma or suspicious circumstances in drowning of Obamas’ chef, police say',
            'output': 'gradcam_wo_visualization5.png',
            'model_path': '/home/shunlizhang/zwk/CMFQN/output/ikcest/womcvnew_q0.8_e_11_0.8408521303258145.pth',
            'lang': 'en'
        }, 
    ]
    
    for params in wo_arr:
        main(**params)