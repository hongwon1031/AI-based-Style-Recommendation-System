# AI-체형보완 스타일 추천 시스템
본 프로젝트는 사용자의 체형을 분석하고, 이에 적합한 패션 스타일을 추천해 주는 웹 애플리케이션입니다.
딥러닝 기반의 인체 분할(Deeplabv3), 체형 분류(CNN) 모델을 사용하여 사용자 이미지를 분석하고,
HR-VTON(High-Resolution Virtual Try-On) 기법을 통해 가상 피팅을 제공합니다.




<img src="/포스터.jpg" width="1000" height="1300">
<img src="/부록.png" width="1636" height="458">

# How to use
- install the package('https://github.com/hongwon1031/pro/blob/master/requirements.txt')
- download pretrained CNN model 'https://drive.google.com/file/d/1nkcS0pXqqoy2PGaLDLAa16Cx5rzmA1n4/view?usp=sharing'
- setting the file path(app2.py, main.py)
- install the app2.py
# image path
- segmented image(cloth)
  - Graphonomy-master/HR-VITON-main/test/test/cloth
- mask image(cloth)
  - Graphonomy-master/HR-VITON-main/test/test/cloth-mask
- segmentation(cloth)
  - Graphonomy-master/HR-VITON-main/test/test/image
- Semantic Segmentation
  - Graphonomy-master/HR-VITON-main/test/test/image-densepose
- Semantic Segmentation(grayscale)
  - Graphonomy-master/HR-VITON-main/test/test/image-parse-v3
- Output Image
  - Graphonomy-master/HR-VITON-main/Output
  - <img src="/HR-VITON-main/Output/output_00000.png" width="300" height="400">

# References
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

Viton Model
https://github.com/lastdefiance20/TryYours-Virtual-Try-On
