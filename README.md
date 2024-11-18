# 합성데이터 기반 객체 탐지 AI 경진대회

# 🏆 Result
## **최종 1등(🏆)**

<img width="70%" src="https://github.com/jjuhyeok/Visol-Synthetic_Image_Object_Detection/assets/49608953/b305bb1b-c966-4a74-b98c-d01f17bc2df2"/>



<img width="100%" src="https://github.com/jjuhyeok/Visol-Synthetic_Image_Object_Detection/assets/49608953/f7849d76-3276-445d-9154-49b3c3c4e95b"/>




주최 : Visol

규모 : 1/1500



===========================================================================
[솔루션 PPT Link](https://github.com/jjuhyeok/Visol-Synthetic_Image_Object_Detection/blob/main/2%EC%B0%A8%ED%8F%89%EA%B0%80_PPT/%5B%EC%A5%AC%ED%98%81%EC%9D%B4%ED%8C%80%5D%5B2%EC%B0%A8%ED%8F%89%EA%B0%80PPT%5D.pdf)  
  
    
  
---

##대회 접근법**

###목표**
- **목표**: 자동차 객체를 인식하고 분류하는 문제 해결. 특히 **합성 데이터**와 **실제 데이터** 간의 차이를 극복하는 것이 핵심

---

### **1. EDA**
#### **도메인 지식 활용 및 문제 발견**
- **쉐보레 스파크 분석**:
  - 학습 데이터에서 **2번 클래스**로 라벨링된 차량은 실제로 1세대 스파크와 유사했습니다. 하지만, 라벨은 2016~2021년형으로 표기되어, **Label Miss Match** 문제가 발생했습니다.
- **IONIQ 분석**:
  - 학습 데이터에는 **IONIQ 하이브리드 2019**만 포함되어 있었지만, 실제로는 다양한 연식과 라인업(일렉트릭 포함)이 존재합니다.
- **티볼리 분석**:
  - 학습 데이터에는 **티볼리 2020년식**만 존재하며, 티볼리 아머, 에어 등 다양한 모델이 포함되지 않았습니다.

#### **t-SNE 분석 및 Cosine Similarity 측정**
- **t-SNE 그래프**:
  - 고차원 특징 데이터를 2차원으로 축소하여 시각화했습니다. 대부분의 클래스는 명확하게 군집을 형성했으나, 일부 클래스는 경계가 모호했습니다.
- **Cosine Similarity 분석**:
  - 유사한 클래스(예: 10번과 14번, 32번과 33번)의 높은 유사도를 확인했습니다. 이로 인해 테스트에서 **분류 오류 가능성**이 높아질 수 있음을 발견하며 이는, 자동차 외부 구조의 특성에 의함을 인식했습니다.

---

### **3. 합성 데이터 활용 및 증강 기법**
#### **다양한 증강 기법**
- **Color Jitter**:
  - Hue, Saturation, Lightness 조정을 통해 다양한 색상 변화를 적용했습니다. 이는 현실에서의 다양한 색상 조합을 반영하기 위함입니다.
- **Brightness & Contrast 조정**:
  - 날씨, 시간 등의 영향을 반영하여 다양한 조명 조건에서의 적응성을 강화했습니다.
- **Equalize Transform**:
  - 합성 이미지의 단순한 텍스처 문제를 보완하고, 다양한 차량의 세부 패턴을 학습할 수 있도록 했습니다.

### **Grad-CAM 분석과 Custom Cutout 기법의 핵심 접근법**

#### **Custom Cutout 기법 도입 배경**
- **Custom Cutout V1**: 
  - **EDA 과정**에서, 합성 이미지에는 차량이 다른 물체나 차량에 의해 가려지는 경우가 없다는 문제를 발견했습니다.
  - 현실 세계에서는 **차량 간 가려짐**이나 **물체에 의해 차량이 부분적으로 가려지는 상황**이 빈번히 발생합니다.
  - 이를 반영하기 위해, **Custom Cutout V1**에서는 차량 이미지의 일부를 가려 현실 세계의 조건을 시뮬레이션함으로써, 가려짐에 대한 적응력을 강화했습니다.

#### **Custom Cutout V2와 V3: Grad-CAM 분석 기반 개선**
- **Grad-CAM 분석 결과**:
  - 모델이 주로 **로고와 헤드램프**에 집중하여 예측을 수행하는 경향이 나타났습니다.
  - 이러한 편향된 집중은 특정 부위에 과도하게 의존하게 되어, 다양한 각도나 조명 조건에서는 **과적합 문제**로 이어질 수 있었습니다.
- **Custom Cutout V2**:
  - 로고와 헤드램프에 대한 집중도를 줄이기 위해, 해당 부위를 더 자주 Cutout하는 방식으로 개선했습니다.
  - 이를 통해 모델이 **다양한 부위**를 고려하도록 유도하여, 편향된 학습을 완화했습니다.
- **Custom Cutout V3**:
  - **Version 3**에서는 더 나아가 프론트 범퍼, DRL, 라디에이터 그릴, 헤드램프 등 **다양한 부위**에 집중할 수 있도록 Cutout 확률을 조정했습니다.
  - 이를 통해 모델이 여러 부위의 특징을 학습하게 하여, **과적합을 줄이고 일반화 성능을 높이는 전략**을 사용했습니다.

#### **Mix-up과 Custom Cutout의 조합 효과**
- **Mix-up 기법**과 **Custom Cutout**을 결합함으로써:
  - 모델의 집중도가 **특정 부위**에 과도하게 쏠리는 문제를 해결했습니다.
  - Grad-CAM 분석에서 다양한 부위(헤드램프, 포그램프, 보닛 등)를 고려하는 예측 패턴이 나타나, **분류 성능과 일반화 능력**이 크게 향상되었습니다.

- **Ablation Study 결과**:
  - Mix-up과 Custom Cutout 적용 후, **mAP 기준으로 약 3%의 성능 향상**을 확인할 수 있었습니다.
  - Grad-CAM 시각화에서도, Cutout 적용 전후의 모델 예측 패턴이 더 넓고 다양한 부위를 고려하는 것으로 나타났습니다.

---




### **4. 모델 검증 전략**
#### **Validation Set 차별화**
- 합성 데이터로 학습한 모델이 실제 이미지를 잘 인식할 수 있도록, Validation Set에 **다양한 Augmentation**을 적용했습니다.
- **Fog 조건 추가**:
  - 안개 등의 날씨 조건을 반영하여, 모델의 일반화 성능을 강화했습니다.
- **Blur 효과 추가**:
  - 움직임이나 초점 문제로 인한 흐릿한 이미지를 반영해, 현실 세계의 다양한 상황에 적응할 수 있도록 했습니다.

---

### **5. 모델 선정 및 실험 관리**
#### **Backbone 모델 선정**
- 다양한 Backbone 모델을 실험한 결과, **ResNeSt-200**이 가장 우수한 성능을 보였습니다.
- **ResNeSt**:
  - Split-Attention 메커니즘을 통해 다양한 차량 속성을 효과적으로 학습했습니다.
  - Stochastic Depth 기법으로 overfitting을 방지하고 일반화 성능을 강화했습니다.

#### **Cascade R-CNN 사용**
- Cascade R-CNN은 기존 Faster R-CNN의 단점을 보완하며, 높은 IoU Threshold를 적용하여 성능을 최적화했습니다.

---

### **6. 성능 분석 및 결과**
#### **Grad-CAM 분석**
- 모델이 주로 로고와 그릴을 보고 예측하는 경향이 있음을 발견했습니다.
- Mixup과 Custom Cutout을 통해 모델이 다양한 부위(헤드램프, 포그램프, 보닛 등)를 고려하도록 개선했습니다.

#### **Ablation Study 결과**
- 다양한 증강 기법과 Custom Cutout 적용으로 성능이 크게 향상되었습니다.
- **mAP 기준**으로 약 3%의 성능 향상을 달성했습니다.

---

### **7. 결론 및 향후 방향**
- 합성 데이터와 실제 이미지 간 차이를 줄이기 위해 다양한 증강 기법과 검증 전략을 도입했습니다.
- **Cascade R-CNN**과 **ResNeSt**의 조합으로 최고의 성능을 달성했으며, 다양한 환경에서도 높은 적응력을 보여줬습니다.
- 향후 더 다양한 차량과 환경을 고려한 데이터 확장이 필요하며, 이 경험이 AI 모델의 실제 적용 사례에도 도움이 되기를 기대합니다.

---



✏️
## **대회 느낀점**
 - 그동안 정형 데이터에만 관심이 많아 정형 데이터 관련 대회만 참여하기 바빠 비전 대회는 하고 있지 않았다.
   하지만 이번 비전 대회는 그 전부터 꾸준히 함께 대회 같이 참여하시는 분께서 같이 나가보자고 하셔서 도전해보게 되었다.
   이번 대회는 객체탐지 대회지만 학습 데이터가 일반적인 현실 사진이 아닌, AI 가 만든 Synthetic 데이터였고, 예측은 실제 이미지로 진행하는 Task였다.
   그렇기에 AI가 만든 이미지라는 특성을 이용해서 모델링을 하는 것이 필요하였다.
   

   
===========================================================================

### 합성 데이터를 활용한 자동차 탐지 AI 모델 개발
- 합성데이터란 실제 환경에서 수집되거나 측정되는 것이 아니라 디지털 환경에서 생성되는 데이터셋으로,  최근 방대한 양질의 데이터셋이 필요해짐에 따라 그 중요성이 대두되고 있습니다.

- 합성 데이터는 데이터 라벨링 작업을 위한 2배 이상의 시간 절약과 10배 가까운 비용을 절감하게 하고, 자동화를 바탕으로 정확한 라벨링의 데이터 그리고 정확한 AI 모델 개발을 위한 데이터의 다양화를 가능하게 합니다.
- 학습용 합성데이터를 활용하여 `자동차 탐지를 수행하는 AI 모델`을 개발해야 합니다.


## Project structure
```
Visol-Synthetic_Image_Object_Detection
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
