# A Cross-Modal Feature Quantization Network for Fake News Detection with Contradiction Information

## Introduction
The rapid growth of social media has led to a
significant increase in highly deceptive multi-modal fake news,
leading to serious negative effects on society. Multi-modal fake
news often contains both obvious and hidden contradictions
between its various elements. However, such cross-modal con-
flicting information remains markedly difficult to capture and
utilize effectively. To address the issue, we propose a cross-modal
feature quantization network (CMFQN) to detect fake news by
**extracting contradiction information from multi-modal data to quantify conflict values**.

![CMFQN Figure](figures/main.png)

## Dataset
The meta-data of the Weibo and IKCEST datasets used in our experiments are available.
* Weibo: ["MRML: Multimodal Rumor Detection by Deep Metric Learning"](https://github.com/plw-study/MRML?tab=readme-ov-file).
* IKCEST: The data was published on [Github](https://github.com/THU-BPM/MR2).
You can also download it directly through the link below (https://drive.google.com/file/d/14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4/view?usp=sharing).

Extract the dataset directly to the `data` directory.

## Code
### Pre-training

The pre-training model of MAE (ViT-Base version) can be downloaded from ["Masked Autoencoders: A PyTorch Implementation"](https://github.com/facebookresearch/mae).
The pre-training model of Chinese-CLIP (clip_cn_vit-b-16 version) can be downloaded from ["Chinese-CLIP"](https://github.com/RainZs/Chinese-CLIP/tree/master).
Please place both downloaded model files in the
`process_data` directory.

### Requirements
It is recommended to create an anaconda virtual environment to run the code.
The detailed version of some packages is available in requirements.txt.
You can install all the required packages using the following command:
```
$ conda install --yes --file requirements.txt
```

### Data Preparation
Because the text in the news is in both Chinese and English. So we extract the data features and store them in the local h5 file.

For Weibo-dataset:

```Shell
bash process_weibo.sh 
```

For IKCEST-dataset:

```Shell
bash process_ikcest.sh 
```
After processing is completed, you will see that an h5 file for storing features has been generated in the `data` folder.

### Run
You can run this model through:
```Shell
bash train_weibo.sh 
```
and
```Shell
bash train_ikcest.sh 
```
The training results and the saved parameter files can be seen in the `ouput` folder

### Visualization:
The visualization part in the experiment was implemented through produce Grad-CAM.
![cam Figure](figures/cam.png)
The visual image can be obtained through:
```Shell
python visualization.py 
```
and the specific sample is set in the `visualization.py`

### License
We have been granted permisson to use Weibo IKCEST datasets for academic studies only.
