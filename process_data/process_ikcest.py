import os
import json
import cv2
import pandas as pd
from tqdm import tqdm
import re

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def process_ikcest_file(input_file, output_dir, dataset_type):

    chinese_data = []
    english_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"file {input_file} error！")
            return

    pbar = tqdm(data.items(), desc=f"cope {os.path.basename(input_file)}")
    
    for item_id, item in pbar:
        if item.get('label', 2) == 2:
            continue
        
        content = item.get('caption', '').strip()
        if not content:  
            continue
            
        img_path = item.get('image_path', '')
        if not img_path:
            continue
            
        valid_img = os.path.basename(img_path)
        
        full_img_path = os.path.join(os.path.dirname(input_file), img_path)
        if not os.path.exists(full_img_path):
            continue
        try:
            if cv2.imread(full_img_path) is None:
                continue
        except Exception as e:
            continue
        
        data_item = {
            'images': valid_img,
            'content': content,
            'label': 0 if item['label'] == 1 else 1  
        }
        
        if is_chinese(content):
            chinese_data.append(data_item)
        else:
            english_data.append(data_item)

    def save_dataset(data, lang):
        if data:
            df = pd.DataFrame(data)[['images', 'content', 'label']] 
            output_path = os.path.join(
                output_dir, 
                f"{dataset_type}_{lang}_datasets_news_data.xlsx"  
            )
            df.to_excel(output_path, index=False)
            print(f"file：{output_path} | num：{len(df)}")

    save_dataset(chinese_data, 'cn')
    save_dataset(english_data, 'en')

if __name__ == "__main__":
    CONFIG = {
        "input_files": {
            "train": "/home/shunlizhang/zwk/CMFQN/data/data/dataset_items_train.json", 
            "val": "/home/shunlizhang/zwk/CMFQN/data/data/dataset_items_val.json",
            "test": "/home/shunlizhang/zwk/CMFQN/data/data/dataset_items_test.json"
        },
        "output_dir": "/home/shunlizhang/zwk/CMFQN/data/data" 
    }
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    for dataset_type in ["train", "val", "test"]:
        print(f"\n{'='*30} cope {dataset_type.upper()} datas {'='*30}")
        process_ikcest_file(
            input_file=CONFIG['input_files'][dataset_type],
            output_dir=CONFIG['output_dir'],
            dataset_type=dataset_type
        )