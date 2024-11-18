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

## **Intro**

### **대회 목표 및 평가 산식**
- **평가 산식**: mAP (IoU threshold = 0.85)
  - 이 대회에서는 **정확한 Bounding Box 탐지 능력**이 요구됩니다. 높은 IoU threshold 기준으로 정확한 객체 탐지가 필요합니다.
- **모델 전략**:
  - **R-CNN 모델** 사용: R-CNN은 물체가 있을 법한 영역을 찾은 뒤, 각 영역별로 분류와 조정을 수행하여 **정확한 Bounding Box** 탐지가 가능합니다.
  - 따라서, **R-CNN 모델을 중심으로 한 접근 전략**을 세우는 것이 중요합니다.

---

## **EDA (탐색적 데이터 분석)**

### **EDA 전략**
- **합성 이미지**와 **실제 세계의 이미지** 간의 차이를 이해하고, 이를 극복하기 위한 전략을 구성하는 것이 EDA의 핵심 목표입니다.

### **1. EDA1 - Label Miss Match**
- 학습 데이터에서 **2번 클래스로 라벨링된 스파크 차량**은 2016~2021년형으로 표기되어 있습니다.
- 하지만, 도메인 지식과 EDA 결과, 실제 이미지에서는 **1세대 스파크 차량**과 유사한 것으로 나타났습니다.
- → **라벨링 오류**가 존재하며, 이를 고려한 데이터 분석이 필요합니다.

### **2. EDA2 - Domain Knowledge 1**
- **0번 클래스와 32번 클래스**의 그릴 부분에서 실제 이미지와 합성 이미지 간에 차이가 있음을 확인했습니다.
- → 이 차이는 모델 학습에 영향을 줄 수 있는 요소입니다.

### **3. EDA3 - Domain Knowledge 2**
- **IONIQ 12번 클래스**의 경우, 연식과 하이브리드/일렉트릭에 따라 **4가지 종류**가 존재합니다.
- 하지만, EDA 결과 학습 데이터에는 **IONIQ 하이브리드 2019 모델**만 포함되어 있었습니다.
- → 학습 데이터의 다양성이 부족하다는 점을 발견했습니다.

### **4. EDA4 - Domain Knowledge 3**
- **32번 클래스 (티볼리)**: 티볼리 2016~2020년형은 1세대, 아머, 에어 등 다양한 모델이 존재합니다.
- 하지만 학습 데이터에는 **티볼리 1세대 2020년형**만 포함되어 있었습니다.
- → 다양한 모델을 반영하지 못한 데이터 구조입니다.

### **5. EDA5 - Domain Knowledge 4**
- 같은 연식의 차량이라도 **여러 Line (모델 변형)**이 존재할 수 있으며, 소비자 기호에 따라 디자인이 다를 수 있습니다.
- 학습 데이터에는 **기본 라인**만 존재하지만, 현실 세계에서는 다양한 Line이 존재함을 고려해야 합니다.

### **6. EDA6 - t-SNE 분석**
- **ImageNet으로 사전 학습된 ViT 모델**로 각 차량 클래스의 특징을 추출하고, 이를 **t-SNE 그래프**로 시각화했습니다.
- → 학습하지 않은 상태에서도 가상 이미지의 클래스 군집이 잘 형성되었습니다.
- 하지만, **일부 클래스는 서로 겹치거나 위치가 가까워** 군집화가 제대로 이루어지지 않는 경우도 있었습니다.

### **7. EDA7 - 코사인 유사도 분석**
- **t-SNE 분석**에서 유사한 클래스가 존재한다는 점을 확인했습니다.
- 이를 정량적으로 측정하기 위해, 각 클래스별 500개의 이미지 특징을 평균화하고 **코사인 유사도**를 계산했습니다.
- **코사인 유사도 0.96 이상**인 클래스들 (예: 10번과 14번, 32번과 33번)을 발견했습니다.
- → 이러한 클래스들은 현실 세계에서 분류하기 어려울 가능성이 높으며, **유사한 클래스 간의 분류 전략**이 필요합니다.

### **8. EDA8 - Insight Check**
- **10번과 14번 클래스**:
  - 로고와 그릴, 차체 모양이 비슷합니다. → **포그램프**에 집중하면 구별이 더 쉽습니다.
- **32번과 33번 클래스**:
  - 헤드라이트와 포그램프가 유사합니다. → **라디에이터 그릴의 아래 부분**을 봐야 구별할 수 있습니다.
- → 차량의 클래스 구분을 위해서는 **한 부위만이 아니라, 여러 부위의 특징을 고려하는 것이 중요**하다는 인사이트를 얻었습니다.

### **9. EDA9 - Grad-CAM 분석**
- 모델이 **자동차의 어느 부분을 보고 클래스 예측**을 하는지 알아보기 위해, Grad-CAM 시각화를 진행했습니다.
- **bbox**를 이용해 차량만을 crop한 후, 간단한 분류를 수행했습니다.
- Grad-CAM 분석 결과, 모델이 **로고와 그릴**에 집중하여 예측하는 경향이 있음을 확인했습니다.
- → 로고와 그릴이 예측 시 핵심적인 역할을 한다는 점이 밝혀졌습니다.

### **10. EDA10 - Insight: 설계 방향성**
- EDA를 통해, 차량의 클래스를 예측하기 위해 **로고와 그릴이 중요**하다는 점을 확인했습니다.
- 그러나, **포그램프, 보닛, 헤드라이트** 등 다양한 부위의 특징도 함께 고려해야 **정확한 예측**이 가능하다는 결론을 도출했습니다.
- 이를 바탕으로, **증강 기법 설계**에서 다양한 부위의 특징을 반영하도록 접근하였으며, 최종적으로 테스트에서도 **클래스 간 경계를 명확히 하고 군집화를 강화**하는 것을 목표로 설정했습니다.

---

## **합성 데이터 활용**

### **1. 합성 데이터 활용 전략**
- 이번 대회의 주요 목표 중 하나는 **합성 이미지의 특성을 잘 활용**하는 것입니다.
- 합성 이미지는 정확한 라벨링이 가능하고, 다양한 변형을 통해 데이터 표현력을 높일 수 있습니다.
- 이를 위해 **Data Augmentation 기법**을 사용하여 합성 이미지의 표현력을 다양화하고, 현실 세계의 이미지에 더 가깝게 만들었습니다.

### **2. 합성 데이터 활용 1: Color Jitter**
- **현실 세계**에서는 다양한 색상의 자동차 이미지가 존재하지만, 학습 데이터셋에는 이와 같은 색상 조합이 충분하지 않았습니다.
- 특히, **투톤으로 튜닝된 자동차**의 경우 현실에서는 흔하지만, 합성 이미지에서는 이러한 변형이 적었습니다.
- 이를 보완하기 위해, **Color Jitter**를 사용하여:
  - **Hue (색상)**, **Saturation (채도)**, **Lightness (명도)**를 임의로 변경했습니다.
  - 다양한 색상 변화를 통해 현실적인 색상 조합을 반영하고, 모델의 색상 민감도를 줄이는 효과를 얻었습니다.

### **3. 합성 데이터 활용 2: Brightness & Contrast 조정**
- **도로 상황**에서는 날씨, 시간, 구름 등의 영향으로 인해 밝기와 대조가 끊임없이 변합니다.
- 합성 이미지에서는 이러한 조명 변화를 충분히 반영하지 못할 수 있습니다.
- **Brightness & Contrast 조정**을 통해:
  - **현실 세계의 조명 변화를 시뮬레이션**하고, 다양한 조명 조건에서의 적응성을 높였습니다.
  - 이를 통해, 모델이 더 정확하게 차량을 탐지할 수 있도록 학습시켰습니다.

### **4. 합성 데이터 활용 3: Equalize Transform**
- 합성 이미지에서의 차량은 실제 이미지보다 **단순한 텍스처와 패턴**으로 표현될 수 있습니다.
- 현실 세계에서는 차량의 브랜드, 모델, 연식에 따라 텍스처와 패턴이 크게 다릅니다.
- **Equalize Transform**을 사용하여:
  - 현실 세계 이미지의 **세부 텍스처와 패턴**을 강조하고, 합성 이미지의 단순함을 보완했습니다.
  - 이로 인해, 모델이 다양한 차량의 특징을 더 잘 인식할 수 있도록 도왔습니다.

### **5. 합성 데이터 활용 4: Mixup 기법**
- 학습 데이터는 클래스별로 **500개의 샘플**만 존재해, 데이터가 제한적이었습니다.
- **Mixup 기법**은 두 이미지를 선형적으로 결합하여 새로운 학습 샘플을 생성합니다.
- **Mixup의 주요 효과**:
  - 클래스 간 **경계를 부드럽게 만들어**, 모델이 특정 클래스에 과도하게 확신하지 않도록 합니다.
  - Grad-CAM 분석 결과, Mixup을 사용했을 때 모델이 **자동차 본연의 형태**에 더 집중하는 경향을 보였습니다.
  - 이를 통해 **Decision Boundary가 확장**되어, 다양한 데이터 분포에서 성능이 향상되었습니다.

### **6. 합성 데이터 활용 5: Custom Cutout 기법**
#### **(1) Cutout이 필요한 이유 1: 가려짐 문제 해결**
- **현실 세계**에서는 차량이 다른 차량이나 물체에 의해 부분적으로 가려질 수 있습니다.
- 그러나 **합성 이미지**에서는 완전한 형태의 차량만 존재하여, 가려지는 상황이 반영되지 않았습니다.
- **Custom Cutout V1**:
  - EDA에서 발견된 이 문제를 해결하기 위해, **일부 부위를 Cutout**하여 가려짐을 시뮬레이션했습니다.
  - 이를 통해, 모델이 가려진 차량에서도 적응할 수 있도록 학습했습니다.

#### **(2) Cutout이 필요한 이유 2: 차량 간 차이 문제 해결**
- **2번 클래스**의 경우, 이미지와 이름 간의 불일치가 발견되었습니다.
- 테스트에서는 다양한 연식의 차량이 나타날 가능성이 있어, 모든 연식을 포괄할 수 있는 Cutout 기법이 필요했습니다.
- **Custom Cutout V2**:
  - 차량의 연식에 관계없이 다양한 부위에서 Cutout을 적용하여, 여러 모델의 특징을 학습할 수 있도록 했습니다.

#### **(3) Cutout이 필요한 이유 3: Mixup의 한계 보완**
- **Mixup 기법**만 적용한 후, Grad-CAM 분석 결과 **로고, 라이트, 포그램프**에 집중하지 않는 경향이 나타났습니다.
- 이는 모델이 중요한 부위에 덜 집중하게 되어, 분류 성능에 부정적인 영향을 줄 수 있습니다.
- **Custom Cutout V3**:
  - 다양한 부위(헤드라이트, 포그램프, 그릴, 보닛, 로고)를 확률적으로 Cutout하여, 모델이 여러 부위를 고려할 수 있도록 설계했습니다.
  - Grad-CAM 결과, Cutout 적용 후 모델이 **더 넓은 부위의 특징**을 학습하게 되어 분류 성능이 개선되었습니다.

#### **Custom Cutout 적용 결과**
- **Custom Cutout** 적용 전/후 Grad-CAM 비교 결과:
  - Cutout 적용 전에는 모델이 주로 **로고와 그릴**에 집중했습니다.
  - Cutout 적용 후에는 **헤드램프, 포그램프, 보닛 등 다양한 부위를 반영**하여 예측 정확도가 향상되었습니다.

---


## **모델 검증 전략**

### **모델 검증 전략 1: Synthetic 데이터와 현실 이미지 간의 간극 해결**
- 학습에 사용된 모든 이미지는 **Synthetic 데이터**이지만, 실제 테스트에서는 **현실 세계의 이미지**가 사용됩니다.
- 만약 학습에 사용된 **Augmentation 기법**을 그대로 Validation Set에 적용한다면:
  - Validation Set에서 대부분 잘 예측할 가능성이 높아지며, 이는 모델의 성능을 제대로 평가하지 못할 수 있습니다.
- 이를 해결하기 위해, **현실 세계에서 나타날 수 있는 다양한 Augmentation 기법**을 Validation Set에만 따로 적용했습니다.
- **검증 목표**:
  - 모델이 Validation Set에서 이러한 Augmentation을 잘 구별할 수 있다면, 현실 세계에서 나타나는 다양한 상황에도 적응할 수 있을 것입니다.
  - 이를 통해 모델의 **강인함(Robustness)**에 대한 신뢰를 확보할 수 있다고 판단했습니다.

### **모델 검증 전략 2: Fog (안개) 조건 반영**
- **EDA 결과**:
  - 대부분의 Train 이미지는 **맑은 날씨** 조건에서 촬영된 것으로 나타났습니다.
  - 그러나 현실 세계에서는 **안개(Fog)**, 비 등의 다양한 날씨 조건이 존재합니다.
- **Fog 조건 적용**:
  - 실제 세계에서의 다양한 날씨 조건에 대한 모델의 **적응성**을 평가하기 위해, Validation Set에 **Fog 효과**를 추가했습니다.
  - 이 Augmentation은 모델이 학습하지 않은 조건이므로, 새로운 상황에서의 일반화 성능을 테스트할 수 있습니다.
- **검증 효과**:
  - Fog 조건을 추가함으로써, 모델의 **일반화 능력**을 강화할 수 있었습니다.
  - 새로운 상황에 대한 검증 신뢰도가 상승했으며, 현실적인 적용 가능성을 높였습니다.

### **모델 검증 전략 3: Blur (모션 블러) 조건 반영**
- **EDA 결과**:
  - 대부분의 Train 이미지는 **선명한 상태**로 촬영된 특징이 있었습니다.
  - 이는 주로 **멈춰 있는 자동차**를 촬영했기 때문에 나타나는 특성입니다.
- **Blur 효과 적용**:
  - 현실 세계에서는 차량의 움직임으로 인한 **모션 블러(Motion Blur)**나 카메라 초점이 흐려지는 상황이 자주 발생합니다.
  - 이러한 다양한 상황에 대한 모델의 적응성을 평가하기 위해, **Blur 효과**를 Validation Set에 추가했습니다.
- **검증 효과**:
  - Blur 조건을 통해 현실적인 상황을 반영함으로써, 모델이 다양한 카메라 설정과 움직임에도 적응할 수 있는지 평가할 수 있었습니다.
  - 이로 인해, **현실 세계에서의 예측 성능과 강인함**을 높일 수 있었습니다.

---



## **실험 관리**

### **1. 실험 관리 1: 모델 선정 과정**
- 다양한 모델을 실험하기 위해, **mmdetection 라이브러리**에서 사용할 수 있는 대부분의 모델을 테스트했습니다.
- **최종 모델 선정**:
  - 여러 모델의 성능을 비교한 결과, **Cascade R-CNN**을 최종 모델로 선정했습니다.
  - Cascade R-CNN은 **정확한 Bounding Box 탐지**와 **높은 IoU threshold**에서의 성능이 우수하여 선택되었습니다.

### **2. 실험 관리 2: Backbone 모델 선정**
- **Backbone 모델 실험**:
  - Transformer 계열과 CNN 계열의 Backbone 모델을 모두 실험했습니다.
  - 실험 결과, **CNN 계열 모델**이 더 나은 성능을 보였으며, 특히 **ResNet 계열**이 우수한 성능을 기록했습니다.
- **최종 Backbone 모델 선정**:
  - **ResNeSt-200**을 최종 Backbone 모델로 선정했습니다.
  - **ResNeSt의 특징**:
    - **Split-Attention 메커니즘**: 다양한 채널 간의 상호작용을 효과적으로 처리하여, 다양한 차량 속성(외관, 크기, 색상 등)을 학습할 수 있습니다.
    - **Stochastic Depth 메커니즘**: 학습 데이터의 다양성으로 인해 발생할 수 있는 **overfitting 문제**를 완화하고, 모델의 일반화 성능을 높이는 데 기여합니다.

### **3. 실험 관리 3: Augmentation 기법 실험**
- 다양한 **Augmentation 기법**을 실험하여 최적의 조합을 찾았습니다.
- **Custom Cutout 실험**:
  - **Version 1**: 물체 간 가려짐 문제를 해결하기 위해, Cutout을 적용하여 가려진 차량 상황을 시뮬레이션했습니다.
  - **Version 2**: Grad-CAM 분석에서 모델이 **로고와 헤드램프**에 집중하는 문제를 발견하고, 해당 부위에 대한 Cutout 빈도를 높여 집중도를 분산시켰습니다.
  - **Version 3**: 다양한 부위(프론트 범퍼, DRL, 라디에이터 그릴 등)에서 Cutout을 적용하여, **모델의 과적합 문제를 줄이고** 일반화 성능을 향상시켰습니다.

### **4. 실험 관리 4: Cascade R-CNN 선정 이유**
- **Cascade R-CNN**은 기존의 **Faster R-CNN**에서의 한계를 극복한 모델입니다.
- **성능 개선 요소**:
  - Faster R-CNN은 높은 IoU threshold에서 **Positive Sample(객체)**의 수가 부족해 성능이 떨어지는 문제가 있었습니다.
  - **Cascade 구조**:
    - Head 부분을 **Cascade로 연결**하여, 점진적으로 높은 IoU Threshold를 적용합니다.
    - 이를 통해, 다양한 IoU threshold에 대해 최적화가 가능해졌습니다.
    - **정확한 Bounding Box** 탐지와 높은 IoU에서의 성능을 동시에 확보할 수 있었습니다.

### **5. 실험 관리 5: ResNeSt의 특징과 장점**
- **ResNeSt Backbone 모델**을 사용한 이유:
  - **차량 속성의 다양성 학습**:
    - 차량 모델 간에는 다양한 외관, 크기, 색상 등이 존재합니다. ResNeSt의 **Split-Attention 메커니즘**은 이러한 다양한 채널 간 상호작용을 효과적으로 학습하여, 차량 속성의 변화를 잘 반영합니다.
  - **다양한 도로 상황 처리**:
    - 현실 세계에서는 다양한 차량 유형과 도로 상황, 조명 조건이 존재합니다.
    - ResNeSt의 **Stochastic Depth 메커니즘**은 overfitting 문제를 완화하고, 다양한 환경에서도 모델의 일반화 성능을 향상시킵니다.

### **6. 실험 관리 6: Custom Cutout의 Grad-CAM 분석 결과**
- **Grad-CAM 분석을 통한 Custom Cutout 검증**:
  - **Version 1**: 물체 간 가려짐 문제를 해결하기 위해 Cutout을 적용했습니다. 이를 통해, 가려진 차량 상황에서도 예측 성능이 개선되었습니다.
  - **Version 2**: Grad-CAM 분석에서 **로고와 헤드램프**에 집중하는 경향을 발견하고, 해당 부위에 대한 Cutout을 적용해 집중도를 분산시켰습니다.
  - **Version 3**: 다양한 부위(보닛, 헤드램프, 포그램프 등)에 대한 Cutout을 적용하여, 가장 다양한 부위에서 특징을 학습할 수 있도록 최적화했습니다.
  - **Grad-CAM 결과**:
    - Custom Cutout 적용 전에는 모델이 주로 **로고와 그릴**에 집중했습니다.
    - Custom Cutout 적용 후에는 다양한 부위에서 특징을 학습하여, **과적합 문제를 줄이고** 더 균형 잡힌 예측이 가능해졌습니다.

---




=============ㅁㄴㅇㅁㄴ=ㅇㅁㄴ=ㅇㅁㄴ=ㅇ=ㅁㄴ=ㅇㅁ=ㄴ=ㅇㄴㅁ=ㅇ=ㅁㄴ=ㅇ=
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
