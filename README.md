# Semantic Segmentation Evaluation on popular benchmarks
This is an all-in-one, pytorch-based framework for training and evaluation of state-of-the-art architectures on
different open-source datasets and benchmarks.

## Supported Datasets
- Cityscapes: 5000 images of urban scenes with high quality annotations (19 training classes) (https://www.cityscapes-dataset.com/)
- CamVid: The Cambridge-driving Labeled Video Database with complete metadata (32 classes) (http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)

## Supported Models
- U-Net (2015) (https://arxiv.org/abs/1505.04597)
- Fast-SCNN (2018) (https://arxiv.org/abs/1902.04502)
- HRNetV2 (2019) (https://arxiv.org/abs/1904.04514)
- BiSeNetV2(ResNet18, 101) (2020) (https://arxiv.org/abs/2004.02147)
- SFNet(ResNet18, STDC1, STDC2) (2020) (https://arxiv.org/abs/2002.10120)
- DDRNet-23, -23-slim, -39 (2021) (https://arxiv.org/abs/2101.06085)
- RegSeg, -Large (2021) (https://arxiv.org/abs/2111.09957)
- STDCNet1, 2 (2021) (https://arxiv.org/abs/2104.13188)
- PIDNet-S, -M, -L (2022) (https://arxiv.org/abs/2206.02066)
- PP-LiteSeg-B, T (2022) (https://arxiv.org/abs/2204.02681)

## Environment
The code is developed under the following configurations:
- Hardware: >= 1 GPU for both training and inference
- Software: Windows, ***CUDA>=10.2, Python>=3.9, PyTorch>=1.8***
- Dependencies: numpy, opencv, pandas, scipy, sklearn, tensorboard, tensorboardX, tqdm
