import os
import cv2
import pandas as pd
from tqdm import tqdm

def process_dataset_group(input_files, image_dirs, output_path):

    final_data = []

    for label, (txt_path, img_dir) in enumerate(zip(input_files, image_dirs)):
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)
            
            with tqdm(total=total_lines//3, desc=f"cope {os.path.basename(txt_path)}", unit='pieces') as pbar:
                while True:
                    metadata = f.readline().strip()
                    image_line = f.readline().strip()
                    content = f.readline().strip()

                    if not metadata:
                        break

                    if not content or content.lower() == 'null':
                        pbar.update(1)
                        continue

                    valid_img = None
                    for url in image_line.split('|'):
                        if url.startswith(('http://', 'https://')):
                            img_name = os.path.basename(url.split('?')[0])
                            img_path = os.path.join(img_dir, img_name)
                            if os.path.exists(img_path):
                                try:
                                    if cv2.imread(img_path) is not None:
                                        valid_img = img_name
                                        break
                                except:
                                    continue

                    if valid_img:
                        final_data.append({
                            'images': valid_img,
                            'content': content,
                            'label': label
                        })

                    pbar.update(1)

    df = pd.DataFrame(final_data)
    df.to_excel(output_path, index=False)
    print(f"\n{os.path.basename(output_path)} Generation completed! Valid data: {len(df)} entries")

if __name__ == "__main__":
    DATA_CONFIG = {
        "train": {
            "input_files": [
                 "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/tweets/train_rumor.txt",   
                 "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/tweets/train_nonrumor.txt" 
            ],
            "image_dirs": [
                "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/rumor_images",    
                "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/nonrumor_images" 
            ],
            "output": "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/train_datasets_news_data.xlsx"
        },
        "test": {
            "input_files": [
                 "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/tweets/test_rumor.txt",   
                "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/tweets/test_nonrumor.txt" 
            ],
            "image_dirs": [
                "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/rumor_images",    
                "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/nonrumor_images"  
            ],
            "output": "/home/shunlizhang/zwk/CMFQN/data/weibo_dataset/test_datasets_news_data.xlsx"
        }
    }


    for dataset_type in ["train", "test"]:
        config = DATA_CONFIG[dataset_type]
        print(f"\n{'='*30} cope {dataset_type.upper()} datas {'='*30}")
        process_dataset_group(
            input_files=config["input_files"],
            image_dirs=config["image_dirs"],
            output_path=config["output"]
        )