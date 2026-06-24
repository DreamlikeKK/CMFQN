import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPImageProcessor, CLIPProcessor
import models_mae
import os
from torch.utils.data import DataLoader
from torch.autograd import Variable
import h5py

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
token_chinese = BertTokenizer.from_pretrained('bert-base-chinese', )

CN_CLIP_MODEL_NAME = "ViT-B-16"
CN_CLIP_DOWNLOAD_ROOT = "/home/shunlizhang/zwk/CMFQN/process_data"

GT_size = 224
word_token_length = 197
image_token_length = 197

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.cuda.set_device(1)

WEIBO21_H5_DIR = "/home/shunlizhang/zwk/CMFQN/data/weibo21_dataset_local"


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def collate_fn_chinese(data):
    sents = [i[0][0] for i in data]
    image = [i[0][1] for i in data]
    image_aug = [i[0][2] for i in data]
    labels = [i[0][3] for i in data]
    category = [0 for i in data]
    GT_path = [i[1] for i in data]
    idx = [i[0][5] for i in data]
    image_clip = [i[0][6] for i in data]

    bert_token_data = token_chinese.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        padding='max_length',
        max_length=word_token_length,
        return_tensors='pt',
        return_length=True
    )

    import cn_clip.clip as clip
    text_inputs = clip.tokenize(sents, context_length=64)
    input_ids1 = text_inputs
    attention_mask1 = (input_ids1 != 0).long()

    input_ids = bert_token_data['input_ids']
    attention_mask = bert_token_data['attention_mask']
    token_type_ids = bert_token_data['token_type_ids']

    image = torch.stack(image)
    image_aug = torch.stack(image_aug)
    labels = torch.LongTensor(labels)
    category = torch.LongTensor(category)
    image_clip = torch.stack(image_clip)

    return (input_ids, attention_mask, token_type_ids, idx, image_clip), \
           (image, image_aug, labels, category, sents), GT_path, (input_ids1, attention_mask1)


class PRE_Net(nn.Module):
    def __init__(self, lang='cn', model_size='base'):
        super().__init__()
        self.lang = lang
        self.image_model = models_mae.__dict__[f"mae_vit_{model_size}_patch16"](norm_pix_loss=False)
        model_path = os.path.join(os.getcwd(), f'process_data/mae_pretrain_vit_{model_size}.pth')
        checkpoint = torch.load(model_path, map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)

        self.text_model = BertModel.from_pretrained('bert-base-chinese', )

        import cn_clip.clip as clip
        self.clip_model, self.clip_preprocess = clip.load_from_name(
            name=CN_CLIP_MODEL_NAME,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            download_root=CN_CLIP_DOWNLOAD_ROOT
        )

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, input_ids, attention_mask, token_type_ids, image, input_ids1, attention_mask1, image_clip):
        device = next(self.parameters()).device

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image = image.to(device)

        with torch.no_grad():
            clip_image_features = self.clip_model.encode_image(image_clip.to(device))
            clip_text_features = self.clip_model.encode_text(input_ids1.to(device))
            clip_image_features = clip_image_features / clip_image_features.norm(dim=-1, keepdim=True)
            clip_text_features = clip_text_features / clip_text_features.norm(dim=-1, keepdim=True)

            mae_features = self.image_model.forward_ying(image)

            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)[0]

        return clip_image_features, clip_text_features, mae_features, text_outputs


if __name__ == "__main__":
    from weibo21_dataset import weibo21_dataset

    model = PRE_Net(lang='cn')
    os.makedirs(WEIBO21_H5_DIR, exist_ok=True)

    for is_train in (True, False):
        h5_filename = "features_dif.h5" if is_train else "val_features_dif.h5"
        h5_path = os.path.join(WEIBO21_H5_DIR, h5_filename)

        dataset = weibo21_dataset(is_train=is_train)
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn_chinese,
            num_workers=4, sampler=None,
            drop_last=is_train, pin_memory=True,
        )

        with h5py.File(h5_path, "w") as h5_file:
            with torch.no_grad():
                for i, items in enumerate(data_loader):
                    texts, others, GT_path, clip_text = items
                    input_ids, attention_mask, token_type_ids, idx, image_clip = texts
                    image, image_aug, labels, category, sents = others
                    input_ids1, attention_mask1 = clip_text

                    input_ids = to_var(input_ids)
                    attention_mask = to_var(attention_mask)
                    image = to_var(image)
                    token_type_ids = to_var(token_type_ids)
                    labels = to_var(labels)
                    input_ids1 = to_var(input_ids1)
                    attention_mask1 = to_var(attention_mask1)

                    clip_image_feat, clip_text_feat, mae_feat, bert_feat = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        image=image,
                        input_ids1=input_ids1,
                        attention_mask1=attention_mask1,
                        image_clip=image_clip
                    )

                    if f"sample_{idx}" in h5_file:
                        print(f"sample_{idx} already exists, skip.")
                        continue

                    sample_group = h5_file.create_group(f"sample_{idx}")
                    sample_group.create_dataset("mae", data=mae_feat.squeeze(0).cpu().numpy(), compression="gzip")
                    sample_group.create_dataset("bert", data=bert_feat.squeeze(0).cpu().numpy(), compression="gzip")
                    sample_group.create_dataset("clip_image", data=clip_image_feat.squeeze(0).cpu().numpy(), compression="gzip")
                    sample_group.create_dataset("clip_text", data=clip_text_feat.squeeze(0).cpu().numpy(), compression="gzip")
                    sample_group.create_dataset("labels", data=labels.cpu().numpy(), compression="gzip")
                    sample_group.create_dataset("idx", data=idx, compression="gzip")
                    sample_group.create_dataset("GT_path", data=GT_path, compression="gzip")

        print(f"Done: {h5_path}")
