#!/bin/bash

# Weibo21 特征抽取：
# 请先将数据集解压到 data/weibo21/，并确保存在
#   train_datasets.xlsx、test_datasets.xlsx
# 以及 rumor_images/、nonrumor_images/ 等图片目录。

echo "Starting set_local_weibo21feat.py ..."
python3 ./process_data/set_local_weibo21feat.py

if [ $? -ne 0 ]; then
    echo "set_local_weibo21feat.py execution failed"
    exit 1
fi
echo "set_local_weibo21feat.py executed successfully"
echo "H5 files saved under data/weibo21_dataset_local/"
