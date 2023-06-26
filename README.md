# 합성데이터 기반 객체 탐지 AI 경진대회

## 🏆 Result
# **Public score 2nd** 0.99647 | **Private score 2nd** 0.99403  
1등과 0.00001차이...

주최 : Visol

주관 : DACON

규모 : LB 2등/1400



===========================================================================
  
  
    
  
  
✏️
## **대회 느낀점**
 - 아이디어를 제일 잘 살렸던 대회라고 생각합니다.
   공정 과정 중에는 여러가지 상황이 존재할 것으로 가정하고 데이터에 접근하였는데
   1. **기계 마모**: 기계가 오래되거나 마모되면, 그 성능이 저하되고, 이로 인해 수집된 데이터의 신뢰성이 떨어질 수 있습니다.
   2. **데이터 일관성 부족**: 공정이 변경되거나, 장비가 교체되거나, 원재료 품질이 변동되는 등의 이유로 데이터 간의 일관성이 떨어질 수 있습니다. 이는 분석을 복잡하게 만들며, 잘못된 결론을 내릴 수 있게 합니다.
   3. **센서 오류 또는 결함**: 기계에서 데이터를 수집하는 주된 방법 중 하나는 센서를 사용하는 것입니다. 센서가 결함이 있거나 오작동하면, 데이터가 왜곡
   4. **환경적 요인**: 환경적 노이즈도 문제가 될 수 있습니다. 예를 들어, 온도, 습도, 기압 등의 변화는 센서의 성능에 영향을 미치고, 이는 데이터에 노이즈를 더하는 결과
  
   위와 같은 이유들로 데이터의 노이즈를 줄이는게 제일 중요한 대회라고 판단을 하여서  
   **kalman filter, moving average, low-pass filter**와 같은 여러 스무딩 기법들을 사용하였고,  
   그 중에서 kalman filter 적용 시 성능이 가장 좋게 상승하였습니다.  
   kalman filter는 '예측'과 '업데이트'라는 두 단계를 반복하는데, 시스템의 동적 모델과 가우시안 노이즈의 통계적 특성을 이용해 '예측' 단계에서는 시스템의 다음 상태를 예측하고, '업데이트' 단계에서는 새로운 측정 데이터를 사용해 이 예측을 조정하기 때문에  
   측정 오차 보정 + 시계열 데이터 특성 + 자기 상관성 요소를 반영하기 위해
   kalman filter를 사용하였습니다.

   또한 이러한 데이터 특성을 활용하기 위해
   일반적인 K-Fold 방식이 아닌, Stratify K-Fold 방법을 사용하였습니다.
   데이터의 각 주기에 class를 부여하여 Stratify 형식으로 학습을 진행하였고
   이 전과 비교해서 성능 향상에 큰 도움을 주었습니다.

   또한 Shapely Value를 활용해서 주최사에게 어떠한 변수들이 제일 중요했는지 알려주었는데
   이를 통해 XAI를 처음 접하게 되어 매우 흥미로웠습니다.

   이번 대회를 통해 데이터에 Domain을 활용할 수 있는 능력을 배웠던 것 같습니다.
   가설을 세우고 그걸 성능과 등수를 통해 검증받는 경험은 정말 짜릿했습니다.
   비록 Data-Leakage로 수상을 하지는 못하였지만
   약 2000여명 중 1등의 예측력을 가지는 모델을 만들었다는 사실에
   다음 대회에서는 꼭 Data-Leakage 이슈에 휘말리지 않고
   수상을 하겠다라고 다짐을 하게 되는 대회였습니다.
   
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
