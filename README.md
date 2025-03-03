# AI-체형보완 스타일 추천 시스템
본 프로젝트는 사용자의 체형을 분석하고, 이에 적합한 패션 스타일을 추천해 주는 웹 애플리케이션입니다.
딥러닝 기반의 인체 분할(Deeplabv3), 체형 분류(CNN) 모델을 사용하여 사용자 이미지를 분석하고,
HR-VTON(High-Resolution Virtual Try-On) 기법을 통해 가상 피팅을 제공합니다.







## 성능 평가
- MobileNet + CNN
  - <img src="/cnn성능평가1.png" width="800" height="321">
  - <img src="/cnn성능평가2.png" width="936" height="176">
  - <img src="/cnn성능평가3.png" width="532" height="102">
- DNN
  - <img src="/dnn성능평가1.png" width="717" height="270">
  - <img src="/dnn성능평가2.png" width="495" height="368">
  - <img src="/dnn성능평가3.png" width="1322" height="362">
    - 오차율 = |((입력 데이터 쓰레기 총량 - 예측 데이터 쓰레기 총량)/입력 데이터 쓰레기 총량)|
    - 오차율 = |((10000 - 11439)/10000)| = 14.39%





HR-VITON
https://github.com/sangyun884/HR-VITON

Posenet
https://github.com/rwightman/posenet-python

Graphonomy
https://github.com/Gaoyiminggithub/Graphonomy

detectron2
https://github.com/facebookresearch/detectron2

cloth image segmentation
https://github.com/ternaus/cloths_segmentation
