# 합성데이터 기반 객체 탐지 AI 경진대회

# 🏆 Result
## - **Public score 2nd** 0.99647 | **Private score 2nd** 0.99403 | **최종 1등(🏆)**

<img width="70%" src="https://github.com/jjuhyeok/Visol-Synthetic_Image_Object_Detection/assets/49608953/f37a3835-8756-4ad7-a77f-cff465f4d323"/>



<img width="100%" src="https://github.com/jjuhyeok/Visol-Synthetic_Image_Object_Detection/assets/49608953/34c7fd28-7117-44f7-8c9e-f092d4a82302"/>




주최 : Visol

규모 : 1/1500



===========================================================================
  
  
    
  
  
✏️
## **대회 느낀점**

   
===========================================================================

### 합성 데이터를 활용한 자동차 탐지 AI 모델 개발
- 합성데이터란 실제 환경에서 수집되거나 측정되는 것이 아니라 디지털 환경에서 생성되는 데이터셋으로,  최근 방대한 양질의 데이터셋이 필요해짐에 따라 그 중요성이 대두되고 있습니다.

- 합성 데이터는 데이터 라벨링 작업을 위한 2배 이상의 시간 절약과 10배 가까운 비용을 절감하게 하고, 자동화를 바탕으로 정확한 라벨링의 데이터 그리고 정확한 AI 모델 개발을 위한 데이터의 다양화를 가능하게 합니다.
- 학습용 합성데이터를 활용하여 `자동차 탐지를 수행하는 AI 모델`을 개발해야 합니다.


## Project structure
```
Synthetic-Data-Object-Detection
├─ .gitignore
├─ archive  # implementation pytorch 
├─ data  
│  ├─ raw
│  │  └─ raw data
│  ├─ ensemble
│  │  └─ ensemble data
│  └─ submission
├─ models
│  └─ model file
├─ mmdetection
│  ├─ configs  
│  │  ├─ _base_  
│  │  └─ visol  # model config
│  │      └─ model config file.py
│  └─ mmdet
├─ mmdetection3.x/
├─ inference.py # Test Inference
├─ grad_cam.py  # GradCam
├─ ensemble.py  # weighted boxes fusion
└─ README.md
```
## Getting Started
`Python 3.8.10` `mmdetection 2.x`
```
git clone https://github.com/Myungbin/Synthetic-Data-Object-Detection.git

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cd mmdetection
python .\tools\train.py .\configs\visol\{config_file.py}
python .\tools\test.py '{config_file_path}' '{model_result_path}' --format-only --out '{result_path}'
```
## Data
[Dataset Link](https://dacon.io/competitions/official/236107/data)  
The project utilizes a custom dataset for training and evaluation.
The dataset consists of labeled images with bounding box annotations for each object of interest.
The dataset is not included in this repository and needs to be prepared separately.


## Experiment
The default augmentations used were `Resize`, `Flip`, and `Normalize`.  
Faster R-CNN, EfficientDet, Swin Transformer, Libra R-CNN, and YOLO were used in the experiment. You can check the other experiments in the "finished_experiments" folder. In the end, `Cascade R-CNN` models was used.  

| Model         | Backbone | Depth | Augmentation                                      |  mAp |
|---------------|----------|-------|---------------------------------------------------|------|
| Cascade R-CNN | SwinT    | -     | Mixup, Cutout                                     | 0.89 |
| Libra R-CNN   | Resnest  | 200   | Mixup, Cutout, AutoAugment, PhotoMetricDistortion | 0.89 |
| Faster R-CNN  | ResNeXt  | 101   | Mixup                                             | 0.91 |
| Faster R-CNN  | ResNeSt  | 200   | Mixup                                             | 0.93 |
| Faster R-CNN  | ResNeSt  | 101   | Mixup, Cutout                                     | 0.95 |
| Cascade R-CNN | ResNeSt  | 200   | Mixup, Cutout, AutoAugment, PhotoMetricDistortion | 0.98 |
| ...           | ...      | ...   | ...                                               | ...  |

#### Ensemble
`Weighted-Boxes-Fusion` on the results of 8 models.

#### pretrain model
You can check the pretrained model weights [google drive](https://drive.google.com/drive/folders/1YaCBzoYmnUIbbKk2q81_x9H-RuYfxUxI?usp=sharing)  

## Result
Public mAp `0.9964`  Private mAP `0.99403`  

To reproduce the results of a public mAP of 0.9964 and a private mAP of 0.99403, follow these steps  
1. [v7.py](https://github.com/Myungbin/Synthetic-Data-Object-Detection/blob/main/mmdetection3.x/configs/visol/v7.py) must be trained and inferred using a seed of **2023** and the **NVIDIA RTX 4090** GPU. 
Additionally, this file utilizes `mmdetection 3.x`, it should be executed based on the following [document](https://mmdetection.readthedocs.io/en/latest/get_started.html).
2. In the [final folder](https://github.com/Myungbin/Synthetic-Data-Object-Detection/tree/main/mmdetection/configs/visol/final), `v3.py` must be trained and inferred using a seed of **1927851590** and the **NVIDIA RTX 3090** GPU.
3. `For all other cases` must be trained and inferred using a seed of **378452678** and the **NVIDIA A100** GPU.  
4. Finally, You must ensemble the inference results of each model using Weighted-Boxes-Fusion. 
The weights for models v1 to v8 are [0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2].
 
You can find detailed information about the experimental results through the PowerPoint presentation.

## Development Environment
```
OS: Window11
CPU: Intel i9-11900K
RAM: 128GB
GPU: NVIDIA GeFocrce RTX3090 & RTX4090 & A100 & V100
```

## Host
- 주최 : 비솔(VISOL)  
- 주관 : 데이콘(DACON)
- 기간 : 2023.05.08 ~ 2023.06.19 09:59  
[Competition Link](https://dacon.io/competitions/official/236107/overview/description)
