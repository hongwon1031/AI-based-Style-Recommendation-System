##load_trained_model의 num_class 4 -> 1
##style -> choice_button으로 수정
##드레스 color 추가
##create_query 함수 수정
##search_fashion_items 함수 수정
from flask import Flask, render_template, request, jsonify, send_from_directory,session
import os
import random
import json
import chardet
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import io
import base64
import tensorflow as tf
import cv2
import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import pickle
import sys
import logging
from tensorflow.keras.losses import MeanAbsoluteError
#%%
app = Flask(__name__)
app.secret_key = 'your_secret_key'
client_id = "6MJp409IkReWzatD385X"
client_secret = "CjbWvFhFrG"
# 모델 로드
URL_down = r'E:\capdi\K-Fashion\URL_download'       #네이버이미지 저장
save_path = r"E:\total3\total3"
os.makedirs(save_path, exist_ok=True)

segmented_save_path = os.path.join(save_path, "static")
os.makedirs(segmented_save_path, exist_ok=True)

#################세현 모델########################
body_model_path = r"E:\total3\total3\cap_model_v600.h5"
custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
body_model = load_model(body_model_path, custom_objects=custom_objects)
##########################################################

#####################홍원 모델#########################################
# body_model_path = r"C:\Users\rlatp\Desktop\my_model_5_each.h5"
# custom_objects={'mae': MeanAbsoluteError()}
# body_model = load_model(body_model_path,custom_objects={'mae': MeanAbsoluteError()})
#####################################################################

# ####################찬웅 모델############################################

######################################################################
# 예측 결과 저장 경로
body_save_model = r'E:\capdi\K-Fashion\model\bodyshape_result'
os.makedirs(body_save_model, exist_ok=True)
shape_json_path = r"E:\capdi\K-Fashion\final_result\shape\predicted_shape.json"
#색상
#색상

import cv2
import numpy as np


color_matches = {
    "화이트": ["블랙","스카이블루","실버"],
    "라벤더": ["퍼플", "민트", "스카이블루"],
    "오렌지": ["레드", "블루", "옐로우"],
    "실버": ["블랙", "브라운", "라벤더"],
    "퍼플": ["핑크", "라벤더", "그레이", "네이비"],
    "핑크": ["퍼플", "라벤더", "그레이", "네이비"],
    "블랙": ["화이트", "베이지", "브라운", "레드", "그린"],
    "옐로우": ["블랙", "브라운", "네이비"],
    "블루": ["스카이블루", "그레이", "네이비"],
    "네이비": ["화이트", "베이지", "라벤더", "핑크", "옐로우", "그린"],
    "스카이블루": ["베이지", "그린", "레드", "그레이"],
    "베이지": ["레드", "브라운", "카키"],
    "와인": ["블랙", "베이지", "브라운", "그린"],
    "골드": ["블랙", "옐로우", "네이비"],
    "브라운": ["화이트", "블랙", "레드", "그린"],
    "레드": ["화이트", "블랙", "베이지", "브라운", "그린"],
    "민트": ["베이지", "블랙"],
    "그레이": ["베이지", "브라운", "라벤더", "퍼플"],
    "카키": ["화이트", "블랙", "그린"],
    "그린": ["민트", "카키", "퍼플"],
    "네온": ["블랙"]
}

# 체형 관련 스타일 사전 정의
body_shape_styles = {
    '삼각형': {
        '상의_핏': '오버사이즈',
        '상의_기장': '크롭',
        '원피스_핏': '오버사이즈',
        '아우터_핏': '루즈'
    },
    '역삼각형': {
        '상의_핏': '노멀',
        '하의_핏': '루즈',
        '원피스_핏': '노멀'
    },
    '원형': {
        '상의_핏': '루즈',
        '하의_핏': '벨보텀',
        '원피스_핏': '타이트'
    },
    '직사각형': {
        '상의_핏': '루즈',
        '하의_핏': '스키니',
        '원피스_핏': '루즈'
    },
    '모래시계형': {
        '상의_핏': '타이트',
        '하의_핏': '타이트',
        '원피스_핏': '타이트'
    }
}

lower_body_shape_styles = {
    '일자하체': {
        '하의_핏': '스키니',
        '아우터_핏': '타이트'
    },
    '삼각하체': {
        '하의_핏': '벨보텀',
        '아우터_핏': '오버사이즈'
    },
    '역삼각하체': {
        '하의_핏': '와이드',
        '아우터_핏': '노멀'
    }
}

leg_length_styles = {
    '롱다리': {
        '하의_기장': '미디',
        '원피스_기장': '미디'
    },
    '숏다리': {
        '하의_기장': '맥시',
        '원피스_기장': '맥시'
    }
}


# 체형에 맞는 스타일 조건 추가하는 함수
def add_body_shape_styles(conditions, body_shape, lower_body_shape, leg_length):
    # 상체에 따른 스타일 추가
    if body_shape in body_shape_styles:
        for key, value in body_shape_styles[body_shape].items():
            if key not in conditions:
                conditions[key] = value

    # 하체에 따른 스타일 추가
    if lower_body_shape in lower_body_shape_styles:
        for key, value in lower_body_shape_styles[lower_body_shape].items():
            if key not in conditions:
                conditions[key] = value

    # 다리 길이에 따른 스타일 추가
    if leg_length in leg_length_styles:
        for key, value in leg_length_styles[leg_length].items():
            if key not in conditions:
                conditions[key] = value

    return conditions

def preprocess_image(image_path):
    # 이미지 파일 읽기
    img = Image.open(image_path).convert("RGB")  # 이미지 로드 및 RGB 변환
    img = img.resize((384, 512))  # 크기 조정
    img = np.array(img) / 255.0  # 배열 변환 및 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가 (1, 384, 512, 3)
    return img

# 체형 분류 함수
def classify_shape(A2, A3, A4, D3, I3):
    if (A3 - A4) > 9.1 and (A3 - A2) < 23.0:
        return '삼각형'
    elif (A4 - A3) >= 9.1 and (A4 - A2) < 23.0:
        return '역삼각형'
    elif (A3 - A4) < 9.1 and (A4 - A3) < 9.1 and (A4 - A2) < 23.0 and (A3 - A2) < 25.0:
        return '직사각형'
    elif (A3 - A4) > 5.1 and (A3 - A2) >= 18.0 and (A3 / A2) >= 0.1193:
        return '원형'
    elif (A4 - A3) <= 2.5 and (A3 - A4) < 9.1 and (A4 - A2) >= 23.0 or (A3 - A2) >= 25.0:
        return '모래시계형'
    else:
        return '알 수 없음'


def classify_lower_body(Q3, S3):
    if (Q3 - S3) > 20:
        return '역삼각하체'
    elif (Q3 - S3) <= 20 and (Q3 - S3) >= 15:
        return '일자하체'
    elif (Q3 - S3) <= 15:
        return '삼각하체'


def classify_leg_length(I3, D3):
    if (D3 / I3) >  0.45 :
        return '롱다리'
    else:
        return '숏다리'


# 예측 결과를 파일로 저장하는 함수
def save_prediction_to_file(D3, I3, A4, A2, A3, Q3, S3, shape_result, lower_body_result, leg_length_result):
    file_path = os.path.join(body_save_model, 'prediction_results.txt')
    with open(file_path, 'w') as file:
        file.write(f"상체 체형: {shape_result}\n")
        file.write(f"하체 체형: {lower_body_result}\n")
        file.write(f"다리 길이: {leg_length_result}\n")
        file.write(
            f"키: {D3:.2f}, 샅높이: {I3:.2f}, 가슴둘레: {A4:.2f}, 허리둘레: {A2:.2f}, 엉덩이둘레: {A3:.2f}, 넙다리둘레: {Q3:.2f}, 장딴지둘레: {S3:.2f}")
    return file_path

import cv2
import glob
import numpy as np
import subprocess
from cloths_segmentation.pre_trained_models import create_model


def process_image_with_segmentation(input_image_path, temp_dir="./temp_graphonomy"):
    try:
        # 디렉토리 준비
        os.makedirs(temp_dir, exist_ok=True)

        # 입력 이미지 읽기
        input_image = cv2.imread(input_image_path)
        if input_image is None:
            raise FileNotFoundError(f"Input image not found: {input_image_path}")

        print("성공1: 이미지 로드 완료")

        # 첫 번째 리사이즈: 768x1024로 리사이즈
        resized_image = cv2.resize(input_image, (768, 1024))
        print("성공2: 768x1024로 리사이즈 완료")

        # 두 번째 리사이즈: 384x512로 리사이즈
        small_image = cv2.resize(resized_image, (384, 512))
        print("성공3: 384x512로 리사이즈 완료")

        # 임시 저장 경로
        temp_input_path = os.path.join(temp_dir, "temp_input.jpg")
        temp_output_name = "segmentation"
        # 절대 경로 설정
        temp_output_path = os.path.abspath(r"E:\\total3\\total3\\Graphonomy-master\\output\\segmentation.png")
        cv2.imwrite(temp_input_path, small_image)

        print("성공4: 임시 입력 파일 저장 완료")

        # Graphonomy 실행
        graphonomy_script_path = os.path.join("E:\\total3\\total3\\Graphonomy-master", "exp",
                                              "inference", "inference.py")
        if not os.path.exists(graphonomy_script_path):
            raise FileNotFoundError(f"Graphonomy inference.py not found at: {graphonomy_script_path}")

        # PYTHONPATH 환경 변수 설정
        graphonomy_env = os.environ.copy()
        graphonomy_env["PYTHONPATH"] = "E:\\total3\\total3\\Graphonomy-master"

        temp_input_path = os.path.abspath(os.path.join(temp_dir, "temp_input.jpg"))
        cv2.imwrite(temp_input_path, small_image)

        # Graphonomy 명령어
        graphonomy_command = [
            f"python" ,
            "E:/total3/total3/Graphonomy-master/exp/inference/inference.py",
            "--loadmodel", "E:\\total3\\total3\\Graphonomy-master\\inference.pth",
            "--img_path", temp_input_path,  # 절대 경로 사용
            "--output_path", temp_dir,
            "--output_name", temp_output_name
        ]

        # 파일 존재 확인
        if not os.path.exists(temp_input_path):
            raise FileNotFoundError(f"Temp input image not found at: {temp_input_path}")

        # 실행 디렉토리 변경 및 복구
        graphonomy_dir = "E:\\total3\\total3\\Graphonomy-master"
        current_dir = os.getcwd()
        try:
            os.chdir(graphonomy_dir)
            subprocess.run(graphonomy_command, env=graphonomy_env, check=True)
            print("Graphonomy 실행 완료")
        finally:
            os.chdir(current_dir)
        # 마스크 읽기
        mask_img = cv2.imread(temp_output_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"Mask image not found: {temp_output_path}")

        mask_img = cv2.resize(mask_img, (768, 1024))
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_img = cv2.erode(mask_img, k)

        # 배경 제거
        segmented_image = cv2.bitwise_and(resized_image, resized_image, mask=mask_img)
        segmented_image = np.where(segmented_image == 0, 215, segmented_image)

        # 세그멘테이션 결과 저장
        segmented_path = os.path.join(temp_dir, "segmented.png")
        cv2.imwrite(segmented_path, segmented_image)
        print("성공6: 세그멘테이션 이미지 저장 완료")

        return segmented_path  # 세그멘테이션 결과 반환

    except Exception as e:
        print(f"오류 발생: {e}")
        return None



# 신체 부위 매핑 (RGB -> 신체 부위 이름)
body_part_mapping = {
    (0, 0, 255): '얼굴',
    (0, 85, 85): '하체(골반, 엉덩이, 허벅지 포함)',
    (0, 255, 255): '오른팔',
    (51, 170, 221): '왼팔',
    (85, 51, 0): '몸통',
    (85, 255, 170): '왼쪽 다리(종아리 포함)',
    (170, 255, 85): '오른쪽 다리(종아리 포함)',
    (255, 0, 0): '머리',
    (255, 85, 0): '가슴'
}

# 허용 오차 설정
COLOR_TOLERANCE = 10

# 매핑 데이터 생성 함수
def create_mapping_data_with_tolerance(image_path):
    # 이미지 로드 및 RGB 변환
    image = Image.open(image_path).convert("RGB")
    image_data = np.array(image)

    # 매핑 데이터 초기화
    mapping_data = [0] * len(body_part_mapping)  # 신체 부위 개수만큼 초기화

    # RGB 값 좌표 추출 및 계산
    for i, (color_tuple, body_part) in enumerate(body_part_mapping.items()):
        # ±10 범위 내 RGB 값 좌표 추출
        mask = np.all(np.abs(image_data - color_tuple) <= COLOR_TOLERANCE, axis=-1)
        coordinates = np.argwhere(mask)

        if len(coordinates) > 0:  # 해당 색상의 픽셀이 존재하는 경우만 처리
            # 신체 부위별 계산 수행
            if body_part == '가슴':
                # 가슴둘레 계산: 중앙 Y 좌표에서 가장 작은 X와 가장 큰 X 차이
                y_mid = (np.min(coordinates[:, 0]) + np.max(coordinates[:, 0])) // 2
                x_coords_at_y_mid = coordinates[coordinates[:, 0] == y_mid][:, 1]
                if len(x_coords_at_y_mid) > 0:
                    mapping_data[i] = np.max(x_coords_at_y_mid) - np.min(x_coords_at_y_mid)

            elif body_part == '몸통':
                # 허리둘레 계산: 가장 작은 X와 가장 큰 X 차이
                x_coords = coordinates[:, 1]
                mapping_data[i] = np.max(x_coords) - np.min(x_coords)

            elif body_part == '하체(골반, 엉덩이, 허벅지 포함)':
                # 엉덩이둘레 계산: 가장 작은 X와 가장 큰 X 차이
                x_coords = coordinates[:, 1]
                mapping_data[i] = np.max(x_coords) - np.min(x_coords)

            elif body_part in ['왼쪽 다리(종아리 포함)', '오른쪽 다리(종아리 포함)']:
                # 넙다리둘레 계산: 중앙 Y 좌표에서 가장 작은 X와 가장 큰 X 차이
                y_top = np.min(coordinates[:, 0])
                y_bottom = np.max(coordinates[:, 0])
                y_mid = (y_top + y_bottom) // 2
                x_coords_at_y_mid = coordinates[coordinates[:, 0] == y_mid][:, 1]
                if len(x_coords_at_y_mid) > 0:
                    mapping_data[i] = np.max(x_coords_at_y_mid) - np.min(x_coords_at_y_mid)

            elif body_part == '얼굴':
                # 얼굴의 가로 길이 계산
                x_coords = coordinates[:, 1]
                mapping_data[i] = np.max(x_coords) - np.min(x_coords)

    # 최종 매핑 데이터 반환
    return np.array(mapping_data, dtype=np.float32)

# 매핑 데이터 생성
image_path = "E:/total3/total3/Graphonomy-master/output/segmentation.png"
mapping_data = create_mapping_data_with_tolerance(image_path)
mapping_data = np.expand_dims(mapping_data, axis=0)  # (9,) -> (1, 9)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 클라이언트에서 전달된 이미지 디코딩
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])  # base64로 인코딩된 이미지 디코딩
        image = Image.open(io.BytesIO(image_data)).convert('RGB')  # RGB 변환
        image_np = np.array(image)

        # 2. 임시 디렉토리에 원본 이미지 저장
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        input_image_path = os.path.join(temp_dir, "input_image.jpg")
        cv2.imwrite(input_image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print("원본 이미지 저장 완료")
        #####################3추가
        img = cv2.imread(input_image_path)

        if img is None:
            print(f"이미지를 읽지 못했습니다. 경로를 확인하세요: {input_image_path}")
        else:
            print("원본 이미지 크기:", img.shape)

            # 목표 크기 설정
            target_width = 768
            target_height = 1024

            # 원본 이미지 크기 가져오기
            h, w = img.shape[:2]

            # 가로 중심 계산
            x_center = w // 2

            # 잘라낼 가로 영역 계산 (중앙 기준)
            x_start = max(0, x_center - (target_width // 2))
            x_end = min(w, x_center + (target_width // 2))

            # 이미지 잘라내기
            cropped_img = img[:, x_start:x_end]

            # 세로 크기가 목표 크기와 다를 경우 리사이즈
            if h != target_height:
                cropped_img = cv2.resize(cropped_img, (target_width, target_height))

            # 결과 저장 경로
            output_image_path = os.path.join(temp_dir, "input_image.jpg")

            # 이미지 저장
            result = cv2.imwrite(output_image_path, cropped_img)
            if not result:
                print(f"이미지 저장에 실패했습니다. 경로를 확인하세요: {output_image_path}")
            else:
                print(f"이미지 저장 성공: {output_image_path}, 최종 크기: {cropped_img.shape}")
        ########################
        # 3. Graphonomy를 활용한 전처리 수행
        output_dir = "./output"  # 원하는 출력 경로 설정
        os.makedirs(output_dir, exist_ok=True)
        preprocessed_image_path = process_image_with_segmentation(input_image_path, output_dir)

        if preprocessed_image_path is None:
            return jsonify({'error': "전처리 과정에서 오류가 발생했습니다."}), 500
        print("Graphonomy 전처리 완료")

        # 이미지 경로
        image_path = "E:\\total3\\total3\\Graphonomy-master\\output\\segmentation.png"
        # 이미지 전처리
        input_image = preprocess_image(image_path)
        predictions = body_model.predict([input_image, mapping_data])





        print("Predicted Measurements:", predictions)
        # 복원된 신체 치수 값을 변수에 매핑
        D3, I3, A4, A2, A3, Q3, S3 = predictions[0]

        # 결과 출력
        print(f"D3: {D3:.2f}, I3: {I3:.2f}, A4: {A4:.2f}, A2: {A2:.2f}, A3: {A3:.2f}, Q3: {Q3:.2f}, S3: {S3:.2f}")

        # 6. 예측 결과를 기반으로 체형 분류
        shape_result = classify_shape(A2, A3, A4, D3, I3)
        lower_body_result = classify_lower_body(Q3, S3)
        leg_length_result = classify_leg_length(I3, D3)
        print("체형 분류 완료")

        # 7. 결과 반환
        return jsonify({
            'predictions': pre1.tolist(),
            'shape_result': shape_result,
            'lower_body_result': lower_body_result,
            'leg_length_result': leg_length_result
        })

    except Exception as e:
        print(f"오류 발생: {e}")
        return jsonify({'error': f"오류 발생: {str(e)}"}), 500

@app.route('/search-json-top', methods=['POST'])
def search_json_top():
    return search_json_clothing('상하의')


@app.route('/search-json-bottom', methods=['POST'])
def search_json_bottom():
    return search_json_clothing('상하의아우터')


@app.route('/search-json-dress', methods=['POST'])
def search_json_dress():
    return search_json_clothing('원피스')


cache_file_path = r"E:\capdi\K-Fashion\cached_json_data.pkl"
base_json_dir = r"E:\capdi\K-Fashion\new_label\new_label"
top = bottom = outer = onepiece = 0


def create_cache(cache_file_path, base_json_dir):
    json_files = [os.path.join(base_json_dir, f) for f in os.listdir(base_json_dir) if f.endswith('.json')]
    json_data_list = []
    total_files = len(json_files)

    for idx, json_file in enumerate(json_files):
        with open(json_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            json_data["파일명"] = json_file  # 파일명 추가
            json_data_list.append(json_data)

        # 진행 상황 출력 (한 줄로 업데이트)
        progress = (idx + 1) / total_files * 100
        sys.stdout.write(f"\r캐시 생성 진행 상황: {progress:.2f}% ({idx + 1}/{total_files})")
        sys.stdout.flush()

    # 읽어온 데이터를 캐시에 저장
    with open(cache_file_path, 'wb') as cache_file:
        pickle.dump(json_data_list, cache_file)
    print("\n캐시에 JSON 데이터를 저장했습니다.")

    return json_data_list


def load_or_create_cache():
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as cache_file:
            print("캐시 파일에서 데이터를 로드했습니다.")
            return pickle.load(cache_file)
    else:
        print("캐시 파일이 존재하지 않습니다. 캐시를 생성합니다.")
        return create_cache(base_json_dir, cache_file_path)


# 기존 코드 수정: search_json_clothing 함수에서 캐시 사용


def search_json_clothing(clothing_type):
    #########################################################################################################json 불러오기
    try:
        with open(shape_json_path, 'r', encoding='utf-8') as f:
            shape_data = json.load(f)
            conditions = shape_data.get('conditions')
    except FileNotFoundError:
        return jsonify({'error': "체형 정보 파일을 찾을 수 없습니다."})
    except json.JSONDecodeError:
        return jsonify({'error': "체형 정보 파일을 불러오는 데 실패했습니다."})

    data = request.json
    # conditions = data.get('conditions')

    ##################################################################################수정
    # 캐시된 JSON 데이터 불러오기
    json_data_list = load_or_create_cache()
    best_match_상의 = None
    best_match_하의 = None
    best_match_아우터 = None
    best_match_원피스 = None
    best_score_상의 = -1
    best_score_하의 = -1
    best_score_아우터 = -1
    best_score_원피스 = -1
    match_found = False  # 모든 조건이 매칭되었는지 확인하는 변수
    print("조건:", conditions)
    n = 0

    start_index = random.randint(0, len(json_data_list) - 1)

    for i in range(start_index, start_index + len(json_data_list)):
        json_data = json_data_list[i % len(json_data_list)]
        라벨링 = json_data.get('데이터셋 정보', {}).get('데이터셋 상세설명', {}).get('라벨링', {})

        # clothing_type에 따라 다른 조건 적용
        if clothing_type == '상하의':
            # '상의'만 존재하고 다른 의상 데이터는 없는 경우
            if (
                    라벨링.get('상의', [{}])[0]
                    and not 라벨링.get('하의', [{}])[0]
                    and not 라벨링.get('원피스', [{}])[0]
                    and not 라벨링.get('아우터', [{}])[0]
            ):
                score = 0
                for key, value in conditions.items():
                    if key.startswith("상의"):
                        category, attribute = key.split('_')
                        item = 라벨링.get(category, [{}])[0]
                        if item.get(attribute) == value:
                            score += 1

                if score > best_score_상의:
                    best_score_상의 = score
                    best_match_상의 = json_data
                    print('최고 매칭 데이터 (상의만):', best_match_상의)
                    print('상의만 점수:', score, '\n')
                    print('데이터 순번:', n)

                if score == len(conditions):
                    match_found = True
                    break

            # '하의'만 존재하고 다른 의상 데이터는 없는 경우
            elif (
                    라벨링.get('하의', [{}])[0]
                    and not 라벨링.get('상의', [{}])[0]
                    and not 라벨링.get('원피스', [{}])[0]
                    and not 라벨링.get('아우터', [{}])[0]
            ):
                score = 0
                for key, value in conditions.items():
                    if key.startswith("하의"):
                        category, attribute = key.split('_')
                        item = 라벨링.get(category, [{}])[0]
                        if item.get(attribute) == value:
                            score += 1

                if score > best_score_하의:
                    best_score_하의 = score
                    best_match_하의 = json_data
                    print('최고 매칭 데이터 (하의만):', best_match_하의)
                    print('하의만 점수:', score, '\n')
                    print('데이터 순번:', n)

                if score == len(conditions):
                    match_found = True
                    break

        elif clothing_type == '상하의아우터':
            # 상의 데이터 처리
            if (
                    라벨링.get('상의', [{}])[0]
                    and not 라벨링.get('하의', [{}])[0]
                    and not 라벨링.get('원피스', [{}])[0]
                    and not 라벨링.get('아우터', [{}])[0]
            ):
                score = 0
                for key, value in conditions.items():
                    if key.startswith("상의"):
                        category, attribute = key.split('_')
                        item = 라벨링.get(category, [{}])[0]
                        if item.get(attribute) == value:
                            score += 1

                if score > best_score_상의:
                    best_score_상의 = score
                    best_match_상의 = json_data
                    print('최고 매칭 데이터 (상의만):', best_match_상의)
                    print('상의만 점수:', score, '\n')
                    print('데이터 순번:', n)

                if score == len(conditions):
                    match_found = True
                    break

            # 하의 데이터 처리
            elif (
                    라벨링.get('하의', [{}])[0]
                    and not 라벨링.get('상의', [{}])[0]
                    and not 라벨링.get('원피스', [{}])[0]
                    and not 라벨링.get('아우터', [{}])[0]
            ):
                score = 0
                for key, value in conditions.items():
                    if key.startswith("하의"):
                        category, attribute = key.split('_')
                        item = 라벨링.get(category, [{}])[0]
                        if item.get(attribute) == value:
                            score += 1

                if score > best_score_하의:
                    best_score_하의 = score
                    best_match_하의 = json_data
                    print('최고 매칭 데이터 (하의만):', best_match_하의)
                    print('하의만 점수:', score, '\n')
                    print('데이터 순번:', n)

                if score == len(conditions):
                    match_found = True
                    break

            # 아우터 데이터 처리
            elif (
                    라벨링.get('아우터', [{}])[0]
                    and not 라벨링.get('상의', [{}])[0]
                    and not 라벨링.get('원피스', [{}])[0]
                    and not 라벨링.get('하의', [{}])[0]
            ):
                score = 0
                for key, value in conditions.items():
                    if key.startswith("아우터"):
                        category, attribute = key.split('_')
                        item = 라벨링.get(category, [{}])[0]
                        if item.get(attribute) == value:
                            score += 1

                if score > best_score_아우터:
                    best_score_아우터 = score
                    best_match_아우터 = json_data
                    print('최고 매칭 데이터 (아우터):', best_match_아우터)
                    print('아우터만 점수:', score, '\n')
                    print('데이터 순번:', n)

                if score == len(conditions):
                    match_found = True
                    break

        elif clothing_type == '원피스':
            if (
                    라벨링.get('원피스', [{}])[0]
                    and not 라벨링.get('상의', [{}])[0]
                    and not 라벨링.get('아우터', [{}])[0]
                    and not 라벨링.get('하의', [{}])[0]
            ):
                score = 0
                for key, value in conditions.items():
                    if key.startswith("원피스"):
                        category, attribute = key.split('_')
                        item = 라벨링.get(category, [{}])[0]
                        if item.get(attribute) == value:
                            score += 1

                if score > best_score_원피스:
                    best_score_원피스 = score
                    best_match_원피스 = json_data
                    print('최고 매칭 데이터 (원피스):', best_match_원피스)
                    print('원피스만 점수:', score, '\n')
                    print('데이터 순번:', n)

                if score == len(conditions):
                    match_found = True
                    break

    # 최종 결과 출력
    print("최종 최고 매칭 데이터 (상의만):", best_match_상의)
    print("최종 최고 점수 (상의만):", best_score_상의)
    print("최종 최고 매칭 데이터 (하의만):", best_match_하의)
    print("최종 최고 점수 (하의만):", best_score_하의)
    print("최종 최고 매칭 데이터 (아우터만):", best_match_아우터)
    print("최종 최고 점수 (아우터만):", best_score_아우터)
    print("최종 최고 매칭 데이터 (원피스만):", best_match_원피스)
    print("최종 최고 점수 (원피스만):", best_score_원피스)

    # 조건에 가장 잘 맞는 파일을 반환
    if best_match_상의 or best_match_하의 or best_match_아우터 or best_match_원피스:
        print('성공')

        # 초기화
        이미지정보_data = []
        상의_data = []
        하의_data = []
        아우터_data = []
        드레스_data = []

        # best_match에 따라 필요한 데이터만 추출
        if clothing_type == '상하의':
            # 상의 데이터 처리
            if best_match_상의:
                with open(best_match_상의["파일명"], 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    image_info = json_data.get('이미지 정보', {})
                    image_id = image_info.get('이미지 식별자', None)
                    이미지정보_data.append(image_id)
                    라벨링 = json_data.get('데이터셋 정보', {}).get('데이터셋 상세설명', {}).get('라벨링', {})

                    if '상의' in 라벨링:
                        for 상의 in 라벨링['상의']:
                            상의_data.append({
                                'category': 상의.get('카테고리', None),
                                'fit': 상의.get('핏', None),
                                '기장': 상의.get('기장', None),
                                'color': 상의.get('색상', None)
                            })

            # 하의 데이터 처리
            if best_match_하의:
                with open(best_match_하의["파일명"], 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    라벨링 = json_data.get('데이터셋 정보', {}).get('데이터셋 상세설명', {}).get('라벨링', {})

                    if '하의' in 라벨링:
                        for 하의 in 라벨링['하의']:
                            하의_data.append({
                                'category': 하의.get('카테고리', None),
                                'fit': 하의.get('핏', None),
                                '기장': 하의.get('기장', None),
                                'color': 하의.get('색상', None)
                            })

        elif clothing_type == '상하의아우터':
            # 상의 데이터 처리
            if best_match_상의:
                with open(best_match_상의["파일명"], 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    image_info = json_data.get('이미지 정보', {})
                    image_id = image_info.get('이미지 식별자', None)
                    이미지정보_data.append(image_id)
                    라벨링 = json_data.get('데이터셋 정보', {}).get('데이터셋 상세설명', {}).get('라벨링', {})

                    if '상의' in 라벨링:
                        for 상의 in 라벨링['상의']:
                            상의_data.append({
                                'category': 상의.get('카테고리', None),
                                'fit': 상의.get('핏', None),
                                '기장': 상의.get('기장', None),
                                'color': 상의.get('색상', None)
                            })

            # 하의 데이터 처리
            if best_match_하의:
                with open(best_match_하의["파일명"], 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    라벨링 = json_data.get('데이터셋 정보', {}).get('데이터셋 상세설명', {}).get('라벨링', {})

                    if '하의' in 라벨링:
                        for 하의 in 라벨링['하의']:
                            하의_data.append({
                                'category': 하의.get('카테고리', None),
                                'fit': 하의.get('핏', None),
                                '기장': 하의.get('기장', None),
                                'color': 하의.get('색상', None)
                            })

            # 아우터 데이터 처리
            if best_match_아우터:
                with open(best_match_아우터["파일명"], 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    라벨링 = json_data.get('데이터셋 정보', {}).get('데이터셋 상세설명', {}).get('라벨링', {})

                    if '아우터' in 라벨링:
                        for 아우터 in 라벨링['아우터']:
                            아우터_data.append({
                                'category': 아우터.get('카테고리', None),
                                '핏': 아우터.get('핏', None),
                                '기장': 아우터.get('기장', None),
                                'color': 아우터.get('색상', None)
                            })

        elif clothing_type == '원피스':
            # 원피스 데이터 처리
            if best_match_원피스:
                with open(best_match_원피스["파일명"], 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    image_info = json_data.get('이미지 정보', {})
                    image_id = image_info.get('이미지 식별자', None)
                    이미지정보_data.append(image_id)
                    라벨링 = json_data.get('데이터셋 정보', {}).get('데이터셋 상세설명', {}).get('라벨링', {})

                    if '원피스' in 라벨링:
                        for 원피스 in 라벨링['원피스']:
                            드레스_data.append({
                                'category': 원피스.get('카테고리', None),
                                'fit': 원피스.get('핏', None),
                                '기장': 원피스.get('기장', None),
                                'color': 원피스.get('색상', None)
                            })

        # 데이터 저장 및 반환 준비
        data_to_save = {
            'recommended_json': [bm["파일명"] for bm in [best_match_상의, best_match_하의, best_match_아우터, best_match_원피스] if
                                 bm],
            'conditions': conditions,
            'image_data': 이미지정보_data,
            '상의_data': 상의_data,
            '하의_data': 하의_data,
            '아우터_data': 아우터_data,
            '드레스_data': 드레스_data
        }
        save_path = os.path.join(body_save_model, 'data.json')

        # JSON 파일로 저장
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)

        print(json.dumps(data_to_save, indent=4, ensure_ascii=False))
        return jsonify(data_to_save)

    else:
        return jsonify({'error': "조건에 맞는 데이터를 찾을 수 없습니다."})


# 텍스트 파일에서 체형 정보 불러오기 함수 추가
def load_body_shape_from_txt(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            f.seek(0)
            text = raw_data.decode(encoding)
            lines = text.splitlines()
            body_shape = lines[0].split(": ")[1].strip()
            lower_body_shape = lines[1].split(": ")[1].strip()
            leg_length = lines[2].split(": ")[1].strip()
            return body_shape, lower_body_shape, leg_length
    except FileNotFoundError:
        print(f"파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return None, None, None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_path, num_classes=1):
    model = models.segmentation.deeplabv3_resnet101(pretrained=False)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model


# 이미지 전처리 함수
def segment_preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)


def predict_and_save(model, image_path, save_path):
    print(f"[Debug] Image path received: {image_path}")
    print(f"[Debug] Save path: {save_path}")
    try:
        # Step 1: 입력 이미지 경로 확인
        if not os.path.exists(image_path):
            print(f"[Error] Input image path not found: {image_path}")
            return {"error": "Image path not found"}, 400
        print(f"[Info] Input image path: {image_path}")

        # Step 2: 이미지 전처리
        try:
            image = segment_preprocess_image(image_path)
            print(f"[Info] Image preprocessing completed. Image shape: {image.shape}")
        except Exception as e:
            print(f"[Error] Image preprocessing failed: {e}")
            return {"error": f"Image preprocessing failed: {e}"}, 500

        # Step 3: 모델 예측
        try:
            with torch.no_grad():
                output = model(image)['out']
            print(f"[Info] Model prediction completed. Output shape: {output.shape}")
        except Exception as e:
            print(f"[Error] Model prediction failed: {e}")
            return {"error": f"Model prediction failed: {e}"}, 500

        # Step 4: 마스크 생성
        try:
            pred_mask = (torch.sigmoid(output) > 0.5).squeeze().cpu().numpy()
            print(f"[Info] Prediction mask created. Mask shape: {pred_mask.shape}")
        except Exception as e:
            print(f"[Error] Mask creation failed: {e}")
            return {"error": f"Mask creation failed: {e}"}, 500

        # Step 5: 원본 이미지 로드 및 리사이즈
        try:
            original_image = Image.open(image_path).convert("RGB")
            original_image = original_image.resize((512, 512))
            original_np = np.array(original_image)
            print(f"[Info] Original image loaded and resized. Shape: {original_np.shape}")
        except Exception as e:
            print(f"[Error] Original image processing failed: {e}")
            return {"error": f"Original image processing failed: {e}"}, 500

        # Step 6: 마스크와 원본 이미지를 결합
        try:
            mask_3d = np.stack([pred_mask] * 3, axis=-1)
            masked_image = np.where(mask_3d > 0, original_np, 255).astype(np.uint8)
            print(f"[Info] Mask applied to original image. Masked image shape: {masked_image.shape}")
        except Exception as e:
            print(f"[Error] Mask application failed: {e}")
            return {"error": f"Mask application failed: {e}"}, 500

        # Step 7: 결과 이미지 저장
        try:
            masked_image_pil = Image.fromarray(masked_image)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            masked_image_pil.save(save_path)
            print(f"[Info] Masked image saved at: {save_path}")
        except Exception as e:
            print(f"[Error] Masked image saving failed: {e}")
            return {"error": f"Masked image saving failed: {e}"}, 500

    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
        return {"error": f"Unexpected error: {e}"}, 500


top_model_path = r"E:\capdi\K-Fashion\model\top.pth"
bottom_model_path = r"E:\capdi\K-Fashion\model\bottom.pth"
outer_model_path = r"E:\capdi\K-Fashion\model\outwear.pth"
onepiece_model_path = r"E:\capdi\K-Fashion\model\onepiece.pth"
top_model = load_trained_model(top_model_path)
bottom_model = load_trained_model(bottom_model_path)
outer_model = load_trained_model(outer_model_path)
onepiece_model = load_trained_model(onepiece_model_path)


# 쿼리 생성 함수
def create_query(choice_button):
    ##초기화
    top_query, bottom_query, outer_query, dress_query = None, None, None, None
    top_color_options, bottom_color_options, outer_color_options, dress_color_options = None, None, None, None
    top_category_options, top_fit_options, bottom_category_options, bottom_fit_options = None, None, None, None
    outer_category_options, outer_color_options, dress_category_options, dress_fit_options = None, None, None, None
    for filename in os.listdir(body_save_model):
        if filename.endswith('.json'):  # JSON 파일만 선택
            file_path = os.path.join(body_save_model, 'data.json')

            # JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

                # 상의_data 저장
                if data['상의_data']:
                    top_category_options = data['상의_data'][0].get('category', None)
                    top_fit_options = data['상의_data'][0].get('fit', None)
                    top_color_options = data['상의_data'][0].get('color', None)
                else:
                    top_category_options, top_fit_options, top_color_options = None, None, None

                    # 하의_data 저장
                if data['하의_data']:
                    bottom_category_options = data['하의_data'][0].get('category', None)
                    bottom_fit_options = data['하의_data'][0].get('fit', None)
                    bottom_color_options = data['하의_data'][0].get('color', None)
                else:
                    bottom_category_options, bottom_fit_options, bottom_color_options = None, None, None

                    # 아우터_data 저장
                if data['아우터_data']:
                    outer_category_options = data['아우터_data'][0].get('category', None)
                    outer_color_options = data['아우터_data'][0].get('color', None)
                else:
                    outer_category_options, outer_color_options = None, None

                    # 드레스_data 저장
                if data['드레스_data']:
                    dress_category_options = data['드레스_data'][0].get('category', None)
                    dress_fit_options = data['드레스_data'][0].get('fit', None)
                    dress_color_options = data['드레스_data'][0].get('color', None)
                else:
                    dress_category_options, dress_fit_options = None, None

    print("Top Category:", top_category_options)
    print("Top Fit:", top_fit_options)
    print("Top Color:", top_color_options)
    print("Bottom Category:", bottom_category_options)
    print("Bottom Fit:", bottom_fit_options)
    print("Bottom Color:", bottom_color_options)
    print("Outer Category:", outer_category_options)
    print("Outer Color:", outer_color_options)
    print("Dress Category:", dress_category_options)
    print("Dress Fit:", dress_fit_options)
    print("Dress color:", dress_color_options)
    print("-" * 30)  # 구분선

    queries = []
    if choice_button == 0:  # 상하의 검색
        if top_category_options:
            top_query = '상의' + f"{top_fit_options or ''} {top_category_options or ''} {top_color_options or ''}".strip()
            queries.append(top_query)
        if bottom_category_options:
            bottom_query = '하의' + f"{bottom_fit_options or ''} {bottom_category_options or ''} {bottom_color_options or ''}".strip()
            queries.append(bottom_query)
    elif choice_button == 1:  # 상하의아우터 검색
        if top_category_options:
            top_query = '상의' + f"{top_fit_options or ''} {top_category_options or ''} {top_color_options or ''}".strip()
            queries.append(top_query)
        if bottom_category_options:
            bottom_query = '하의' + f"{bottom_fit_options or ''} {bottom_category_options or ''} {bottom_color_options or ''}".strip()
            queries.append(bottom_query)
        if outer_category_options:
            outer_query = '아우터' + f" {outer_category_options or ''} {outer_color_options or ''}".strip()
            queries.append(outer_query)
    elif choice_button == 2:  # 원피스 검색
        if dress_fit_options and dress_category_options:
            dress_query = '원피스' + f"{dress_fit_options} {dress_category_options} {dress_color_options}"
            queries.append(dress_query)

    elif choice_button == 3:  # 상의만 검색
        if top_category_options:
            top_query = '상의' + f"{top_fit_options or ''} {top_category_options or ''} {top_color_options or ''}".strip()
            queries.append(top_query)
    elif choice_button == 4:  # 하의만 검색
        if bottom_category_options:
            bottom_query = '하의' + f"{bottom_fit_options or ''} {bottom_category_options or ''} {bottom_color_options or ''}".strip()
            queries.append(bottom_query)
    elif choice_button == 5:  # 아우터만 검색
        if outer_category_options:
            outer_query = '아우터' + f" {outer_category_options or ''} {outer_color_options or ''}".strip()
            queries.append(outer_query)

    print("Generated query:", queries)
    print('top query:', top_query)
    print('bottom query:', bottom_query)
    print('outer query:', outer_query)
    print('dress query:', dress_query)

    return (queries,
            top_query, bottom_query, outer_query, dress_query,
            top_category_options, top_fit_options, top_color_options,
            bottom_category_options, bottom_fit_options, bottom_color_options,
            outer_category_options, outer_color_options,
            dress_category_options, dress_fit_options, dress_color_options)

def search_fashion_items(choice_button):
    # 쿼리를 각각 분리하여 가져옵니다.
    queries, top_query, bottom_query, outer_query, dress_query, top_category_options, top_fit_options, top_color_options, bottom_category_options, bottom_fit_options, bottom_color_options, outer_category_options, outer_color_options, dress_category_options, dress_fit_options, dress_color_options = create_query(
        choice_button)

    print("Generated query:", queries)

    # 각각의 쿼리에 대해 개별적으로 결과를 저장할 딕셔너리를 생성합니다.
    results = {
        'top': [],
        'bottom': [],
        'outer': [],
        'dress': []
    }
    seen_titles = set()

    # 모델이 없는 이미지를 판별하는 함수
    def is_clothing_only(image_url):
        try:
            # 이미지를 요청하여 확인
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                # 이미지 크기나 특성을 기반으로 필터링 (예: 특정 배경 색상, 모델 얼굴 감지 등)
                # OpenCV를 이용해 사람 얼굴이 있는지 확인
                image_np = np.array(image)
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                return len(faces) == 0  # 얼굴이 없으면 옷만 있는 것으로 간주
            return False
        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            return False

    # 쿼리별 검색 수행 함수
    def fetch_query_results(query, result_key, color_options):
        if not color_options:
            color_options = []
        start = random.randint(1, 100)  # 각 쿼리마다 랜덤한 시작점 설정
        excluded_titles = ""
        while len(results[result_key]) < 5:  # 각 쿼리에 대해 5개의 결과를 얻을 때까지 반복
            modified_query = f"{query}  {excluded_titles}"
            url = f"https://openapi.naver.com/v1/search/shop.json?query={modified_query}&display=100&start={start}"
            headers = {
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret
            }
            response = requests.get(url, headers=headers)
            print(modified_query)
            print(excluded_titles)
            if response.status_code == 200:
                data = response.json()
                for item in data['items']:
                    title = item['title']
                    price = int(item['lprice'])
                    image_url = item['image']

                    # 가격 범위와 색상 필터 확인
                    if (
                        title not in seen_titles
                        and any(color in title for color in color_options)
                          # 성별 키워드 필터 추가
                    ):
                        # 모델이 없는 이미지인지 확인
                        if is_clothing_only(image_url):
                            results[result_key].append(item)
                            seen_titles.add(title)

                            excluded_titles += f"-{title} "

                            # 이미지 다운로드 및 저장
                            image_response = requests.get(image_url)
                            if image_response.status_code == 200:
                                filename = "".join([c if c.isalnum() else "_" for c in title])
                                image_path = os.path.join(save_path, f"{filename}.jpg")
                                image = Image.open(BytesIO(image_response.content))
                                image = image.convert('RGB')
                                image.save(image_path)
                                print(results, '하하하')
                            # 쿼리별 결과가 5개에 도달하면 다음 쿼리로 이동
                            if len(results[result_key]) == 5:
                                break
                start += 100  # 다음 요청을 위한 시작점 조정
            else:
                print("Error Code:", response.status_code)
                break  # API 오류 시 중단

    # 성별 키워드 설정

    # 각각의 쿼리에 대해 fetch_query_results 함수를 호출
    if top_query:
        fetch_query_results(top_query, 'top', top_color_options)
    if bottom_query:
        fetch_query_results(bottom_query, 'bottom', bottom_color_options)
    if outer_query:
        fetch_query_results(outer_query, 'outer', outer_color_options)
    if dress_query:
        fetch_query_results(dress_query, 'dress', dress_color_options)

    return results


# 기본 페이지
@app.route('/')
def index():
    return render_template('index_3.html')


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    choice_button = int(data['style'])

    # 검색 로직 수행
    results = search_fashion_items(choice_button)

    # 서버 세션에 저장
    session['results'] = results

    # HTML 템플릿을 반환
    return render_template('results_3.html', results=results)


@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        # 디버깅 로거 설정
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('segment_image')

        # 요청 데이터 확인
        data = request.get_json()
        logger.debug(f"Received request data: {data}")

        if not data:
            logger.error("No JSON data provided in the request.")
            return jsonify({"error": "No JSON data provided"}), 400

        # 필수 필드 확인
        image_path = data.get('image_path')
        query = data.get('query')

        if not image_path:
            logger.error("Missing 'image_path' field in the request data.")
            return jsonify({"error": "Missing 'image_path' field"}), 400
        if not query:
            logger.error("Missing 'query' field in the request data.")
            return jsonify({"error": "Missing 'query' field"}), 400

        logger.debug(f"Extracted image_path: {image_path}, query: {query}")

        # URL에서 이미지 다운로드 처리
        if image_path.startswith("http"):
            response = requests.get(image_path)
            if response.status_code == 200:
                filename = os.path.basename(image_path)
                local_image_path = os.path.join(URL_down, filename)
                with open(local_image_path, 'wb') as f:
                    f.write(response.content)
                image_path = local_image_path  # 로컬 경로로 변경
                logger.debug(f"Image downloaded and saved at: {image_path}")
            else:
                logger.error(f"Failed to download image from URL: {image_path}")
                return jsonify({"error": "Failed to download image"}), 400
#시발
        # Segmented 이미지 경로 설정
        segmented_image_path = os.path.join(segmented_save_path, f"cloth_web.jpg")
        logger.debug(f"Segmented image path: {segmented_image_path}")

        # query에 따른 작업 수행
        global top, bottom, outer, dress
        if query == 'top':
            predict_and_save(top_model, image_path, segmented_image_path)
            logger.info(f"Top prediction completed. Saved to: {segmented_image_path}")
        elif query == 'bottom':
            predict_and_save(bottom_model, image_path, segmented_image_path)
            logger.info(f"Bottom prediction completed. Saved to: {segmented_image_path}")
        elif query == 'outer':
            predict_and_save(outer_model, image_path, segmented_image_path)
            logger.info(f"Outer prediction completed. Saved to: {segmented_image_path}")
        elif query == 'onepiece':
            predict_and_save(onepiece_model, image_path, segmented_image_path)
            logger.info(f"Onepiece prediction completed. Saved to: {segmented_image_path}")
        else:
            logger.error(f"Invalid query value: {query}")
            return jsonify({"error": "Invalid query value"}), 400

        return jsonify({"message": "Segmentation completed", "segmented_image_path": segmented_image_path})

    except Exception as e:
        logger.exception(f"Error occurred during segmentation: {e}")
        return jsonify({"error": "Internal server error"}), 500



@app.route('/index3')
def index3():
    return render_template('index3.html')  # templates/index3.html 파일을 렌더링



@app.route('/search-category', methods=['POST'])
def search_category():
    data = request.get_json()
    category = data.get('category')

    choice_map = {'top': 3, 'bottom': 4, 'outer': 5, 'dress': 2}
    choice_button = choice_map.get(category)

    if choice_button is None:
        return jsonify({'error': 'Invalid category'}), 400

    # 기존 results 가져오기
    results = session.get('results', {'top': [], 'bottom': [], 'outer': [], 'dress': []})

    # 특정 카테고리만 업데이트
    new_results = search_fashion_items(choice_button)
    results[category] = new_results[category]

    # 업데이트된 results 다시 세션에 저장
    session['results'] = results

    return jsonify({'results': results})


#################################vton#####################################
data_list = []
import shutil

@app.route('/2', methods=['GET', 'POST'])
def main():
    return render_template('main.html')
@app.route('/buffer', methods=['GET', 'POST'])
def buffer():
    return render_template('buffer.html')
@app.route('/process', methods=['POST', 'GET'])
def process():
    try:
        # 실제 작업을 시뮬레이션 (예: 이미지 처리 및 URL 생성)
        result_image_url = "/static/finalimg.png"
        message = "Virtual fitting completed successfully!"

        # final.html에 데이터 전달
        return render_template('final.html', message=message, image_url=result_image_url)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/fileUpload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        f = request.files['file']
        f_src = 'static/origin_web.jpg'

        f.save(f_src)
        return render_template('fileUpload.html')


@app.route('/fileUpload_cloth_auto', methods=['POST'])
def file_upload_cloth_auto():
    # 지정된 경로에서 최신 파일을 찾기
    directory = r'E:\total3\total3\static'
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    if not files:
        return "No files found in the specified directory.", 400

    latest_file = max(files, key=os.path.getctime)

    # 파일을 지정된 경로로 복사
    destination = 'static/cloth_web.jpg'
    shutil.copy(latest_file, destination)

    return "Latest cloth file uploaded successfully!", 200


@app.route('/view', methods=['GET', 'POST'])
def view():
    print("inference start")

    # main.py 실행
    terminnal_command = "python main.py"
    os.system(terminnal_command)

    print("inference end")

    # 결과 메시지 및 이미지 경로 정의
    message = "Inference completed successfully!"
    result_image_url = "/static/finalimg.png"  # 처리된 이미지의 경로

    # final.html 템플릿 렌더링
    return render_template('final.html', message=message, image_url=result_image_url)


# 저장할 디렉토리 설정
SAVE_DIR = r"E:\total3\total3\save_test"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/save-image', methods=['POST'])
def save_image():
    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"success": False, "error": "No image data provided."}), 400

        # 데이터 URL에서 이미지 데이터 추출
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # 파일 이름 설정
        file_name = os.path.join(SAVE_DIR, "captured_image.jpg")

        # 이미지 파일 저장
        with open(file_name, "wb") as f:
            f.write(image_bytes)

        return jsonify({"success": True, "file_path": file_name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
###############################################################################################

if __name__ == '__main__':
    app.run()
