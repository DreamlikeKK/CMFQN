import copy
import cv2
import torch
import torch.utils.data as data
import data_util as data_util
import albumentations as A
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# 原始数据目录：需包含 train_datasets.xlsx / test_datasets.xlsx 及对应图片
WEIBO21_DATA_ROOT = "/home/shunlizhang/zwk/CMFQN/data/weibo21"


def _resolve_weibo21_image_rel(data_root, img_cell):
    """解析图片相对路径；多图字段取第一张，并按 rumor/nonrumor 目录回退查找。"""
    raw = str(img_cell)
    first = raw.split('|')[0].strip()
    if not first or first.lower() == 'nan':
        return None
    rel_candidates = []
    seen = set()
    for cand in (
        first,
        os.path.basename(first),
        os.path.join('rumor_images', os.path.basename(first)),
        os.path.join('nonrumor_images', os.path.basename(first)),
    ):
        if cand and cand not in seen:
            seen.add(cand)
            rel_candidates.append(cand)
    for rel in rel_candidates:
        full = os.path.join(data_root, rel)
        if os.path.isfile(full):
            return rel
    return None


class weibo21_dataset(data.Dataset):

    def __init__(self, image_size=224, is_train=True):
        super(weibo21_dataset, self).__init__()
        self.is_train = is_train
        self.label_dict = []
        self.image_size = image_size
        self.transform_just_resize = A.Compose(
            [A.Resize(always_apply=True, height=image_size, width=image_size)]
        )
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.CLAHE(always_apply=False, p=0.25),
                    A.RandomBrightnessContrast(always_apply=False, p=0.25),
                    A.Equalize(always_apply=False, p=0.25),
                    A.RGBShift(always_apply=False, p=0.25),
                ]),
                A.OneOf([
                    A.ImageCompression(always_apply=False, quality_lower=60, quality_upper=100, p=0.2),
                    A.GaussianBlur(always_apply=False, p=0.2),
                    A.GaussNoise(always_apply=False, p=0.2),
                    A.ISONoise(always_apply=False, p=0.2)
                ]),
                A.Resize(always_apply=True, height=image_size, width=image_size)
            ]
        )
        from torchvision import transforms
        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        data_root = WEIBO21_DATA_ROOT
        split_file = 'train_datasets.xlsx' if is_train else 'test_datasets.xlsx'
        df = pd.read_excel(os.path.join(data_root, split_file))

        self.data_root = data_root

        for _, row in tqdm(df.iterrows(), desc=f"Loading weibo21 {'train' if is_train else 'test'}"):
            img_rel = _resolve_weibo21_image_rel(data_root, row['image'])
            if img_rel is None:
                continue
            content = str(row['content'])
            if pd.isna(row['content']) or len(content.strip()) == 0:
                continue
            # 原始标签 0=谣言/假、1=非谣言/真；项目内部 1=假、0=真
            raw_label = int(row['label'])
            label = 1 if raw_label == 0 else 0
            self.label_dict.append({
                'images': img_rel,
                'label': label,
                'content': content
            })

    def __getitem__(self, index):
        record = self.label_dict[index]
        images, label, content = record['images'], record['label'], record['content']
        GT_path = os.path.join(self.data_root, images)

        img_GT = cv2.imread(GT_path, cv2.IMREAD_COLOR)
        if img_GT is None:
            img_GT = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        if img_GT.ndim == 2:
            img_GT = np.expand_dims(img_GT, axis=2)
        if img_GT.shape[2] > 3:
            img_GT = img_GT[:, :, :3]

        img_GT = data_util.channel_convert(img_GT.shape[2], 'RGB', [img_GT])[0]

        img_pil = Image.fromarray(cv2.cvtColor(img_GT, cv2.COLOR_BGR2RGB))
        img_clip = self.clip_transform(img_pil)

        if not self.is_train:
            img_GT_augment = self.transform_just_resize(image=copy.deepcopy(img_GT))["image"]
        else:
            img_GT_augment = self.transform(image=copy.deepcopy(img_GT))["image"]

        img_GT = self.transform_just_resize(image=copy.deepcopy(img_GT))["image"]
        img_GT = img_GT.astype(np.float32) / 255.
        img_GT_augment = img_GT_augment.astype(np.float32) / 255.

        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        if img_GT_augment.shape[2] == 3:
            img_GT_augment = img_GT_augment[:, :, [2, 1, 0]]

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_GT_augment = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_augment, (2, 0, 1)))).float()

        return (content, img_GT, img_GT_augment, label, 0, index, img_clip), (GT_path)

    def __len__(self):
        return len(self.label_dict)
