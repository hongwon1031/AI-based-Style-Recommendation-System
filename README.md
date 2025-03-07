# AI-체형보완 스타일 추천 시스템
본 프로젝트는 사용자의 체형을 분석하고, 이에 적합한 패션 스타일을 추천해 주는 웹 애플리케이션입니다.
딥러닝 기반의 인체 분할(Deeplabv3), 체형 분류(CNN) 모델을 사용하여 사용자 이미지를 분석하고,
HR-VTON(High-Resolution Virtual Try-On) 기법을 통해 가상 피팅을 제공합니다.




<img src="/포스터.jpg" width="1000" height="1300">
<img src="/부록.png" width="1636" height="458">

# Stack

<img src="https://img.shields.io/badge/googlecolab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"/> <img src="https://img.shields.io/badge/flask-000000?style=flat-square&logo=flask&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/HTML-E34F26?style=flat-square&logo=HTML5&logoColor=white"/>

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/Tensorflow-43B02A?style=flat-square&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/> <img src="https://img.shields.io/badge/Anaconda-44A833?style=flat-square&logo=anaconda&logoColor=white"/>

# How to use
  - install the package('https://github.com/hongwon1031/pro/blob/master/requirements.txt')
  - download pretrained CNN model 'https://drive.google.com/file/d/1nkcS0pXqqoy2PGaLDLAa16Cx5rzmA1n4/view?usp=sharing'
  - setting the file path(app2.py, main.py)
  - install the app2.py
  <img src="/동작예시1.png" width="800" height="400">
  <img src="/동작예시2.png" width="800" height="400">
  <img src="/동작예시3.jpg" width="800" height="400">

# image path
  - segmented image(cloth)
    - Graphonomy-master/HR-VITON-main/test/test/cloth
    <img src="/HR-VITON-main/test/test/cloth/00001_00.jpg" width="300" height="400">
  - mask image(cloth)
    - Graphonomy-master/HR-VITON-main/test/test/cloth-mask
    <img src="/HR-VITON-main/test/test/cloth-mask/00001_00.jpg" width="300" height="400">
  - Segmentated image(model)
    - Graphonomy-master/HR-VITON-main/test/test/image
    <img src="/HR-VITON-main/test/test/image/00001_00.jpg" width="300" height="400">
  - Semantic Segmentation(model)
    - Graphonomy-master/HR-VITON-main/test/test/image-densepose
    <img src="/HR-VITON-main/test/test/image-densepose/00001_00.jpg" width="300" height="400">
  - Semantic Segmentation_grayscale(model)
    - Graphonomy-master/HR-VITON-main/test/test/image-parse-v3
    <img src="/HR-VITON-main/test/test/image-parse-v3/00001_00.png" width="300" height="400">
  - Output Image
    - Graphonomy-master/HR-VITON-main/Output
    <img src="/HR-VITON-main/Output/output_00000.png" width="300" height="400">
    


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
