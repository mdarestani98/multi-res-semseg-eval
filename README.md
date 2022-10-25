# Semantic Segmentation Evaluation on popular benchmarks

This is an all-in-one, pytorch-based framework for training and evaluation of state-of-the-art architectures on
different open-source datasets and benchmarks.

## Supported Datasets

- [Cityscapes](https://www.cityscapes-dataset.com/): 5000 images of urban scenes with high quality annotations (19 training classes) 
- [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/): The Cambridge-driving Labeled Video Database with complete metadata (32 classes)

## Supported Models

- [U-Net](https://arxiv.org/abs/1505.04597) (2015) 
- [Fast-SCNN](https://arxiv.org/abs/1902.04502) (2018)
- [BiSeNetV1](https://arxiv.org/abs/1808.00897)(ResNet18, 101) (2018)
- [HRNetV2](https://arxiv.org/abs/1904.04514) (2019) 
- [SFNet](https://arxiv.org/abs/2002.10120)(ResNet18, STDC1, STDC2) (2020)
- [DDRNet](https://arxiv.org/abs/2101.06085)-23, -23-slim, -39 (2021) 
- [RegSeg](https://arxiv.org/abs/2111.09957), -Large (2021) 
- [STDCNet](https://arxiv.org/abs/2104.13188)1, 2 (2021) 
- [PIDNet](https://arxiv.org/abs/2206.02066)-S, -M, -L (2022) 
- [PP-LiteSeg](https://arxiv.org/abs/2204.02681)-B, -T (2022) 

## Environment

The code is developed under the following configurations:
- Hardware: >= 1 GPU for both training and inference
- Software: Windows, ***CUDA>=10.2, Python>=3.9, PyTorch>=1.8***
- Dependencies: numpy, opencv, pandas, scipy, sklearn, tensorboard, tensorboardX, tqdm

## Usage

### 0. Prepare datasets

- Download [Cityscapes](https://www.cityscapes-dataset.com/) and 
[CamVid](https://www.kaggle.com/datasets/carlolepelaars/camvid) and unzip them into `data/cityscapes` and `data/CamVid` 
respectively. Run `prepare_data({dataset-dir})` from `dataset/{dataset-name}.py` to parse image files and save path lists.
- 
