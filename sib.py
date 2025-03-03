import numpy as np
from PIL import Image
from scipy.spatial.distance import euclidean
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
# 신체 부위 색상 매핑
body_part_mapping = {
    (0, 0, 255): '얼굴',
    (0, 85, 85): '하체(골반, 엉덩이, 허벅지 포함)',
    (0, 255, 255): '오른팔',  # 시안색
    (51, 170, 221): '왼팔',  # 밝은 하늘색
    (85, 51, 0): '몸통',  # 어두운 갈색
    (85, 255, 170): '왼쪽 다리(종아리 포함)',  # 밝은 연두색
    (170, 255, 85): '오른쪽 다리(종아리 포함)',  # 연두색
    (255, 0, 0): '머리',  # 빨간색
    (255, 85, 0): '가슴'  # 주황색
}

def extract_body_part_features(image_data):
    """세그멘테이션 이미지에서 신체 부위별 특징 추출 (유클리드 거리 기반)"""
    features = []
    unique_colors = np.unique(image_data.reshape(-1, image_data.shape[2]), axis=0)

    for color in unique_colors:
        # 배경 색은 제외
        if tuple(color) == (0, 0, 0):
            continue

        # 색상 매핑 (근사값을 이용한 매칭)
        closest_body_part = '알 수 없는 신체 부위'
        min_distance = float('inf')
        for mapped_color, body_part in body_part_mapping.items():
            dist = euclidean(color, mapped_color)
            if dist < min_distance:
                min_distance = dist
                closest_body_part = body_part

        # 해당 색상의 픽셀 좌표 추출
        mask = np.all(np.abs(image_data - color) <= 10, axis=-1)  # 색상 차이가 10 이하인 픽셀 추출
        coordinates = np.argwhere(mask)

        if len(coordinates) == 0:
            continue

        # 평균 색상 및 크기 (특징)
        mean_color = np.mean(image_data[coordinates[:, 0], coordinates[:, 1]], axis=0)
        area_size = len(coordinates)  # 부위 영역 크기 (픽셀 개수)

        # 신체 부위별 특징 벡터 추가
        features.extend(mean_color)  # 평균 색상
        features.append(area_size)  # 영역 크기

    return features


image_dir = r'E:\body\densepose_images'
csv_dir = r'E:\body\csvfiles'
model_save_path = r'E:\body\model\trained_transformer_model.h5'

# 이미지 및 측정값 저장 리스트
features_list = []
measurements = []

# 이미지와 CSV 파일 이름 정렬
image_names = sorted(os.listdir(image_dir))
csv_names = sorted(os.listdir(csv_dir))

# 이미지와 CSV 매칭 및 처리
for image_name in image_names:
    # 이미지 ID 추출
    image_identifier = os.path.splitext(image_name)[0]  # 확장자 제거
    # 해당하는 CSV 찾기
    matched_csv = next((csv_name for csv_name in csv_names if image_identifier in csv_name), None)
    if not matched_csv:
        continue  # 매칭 실패 시 건너뜀

    # 이미지 로드
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)
    image_data = np.array(image)

    # 신체 부위별 특징 추출
    features = extract_body_part_features(image_data)

    # CSV에서 실제 신체 치수 데이터 읽기
    csv_path = os.path.join(csv_dir, matched_csv)
    csv_data = pd.read_csv(csv_path, encoding='latin-1')

    # 지정된 셀 번호의 데이터만 읽기
    csv_measurements = csv_data.iloc[1:, [3, 8, 12, 13, 15, 16, 18]].values.astype(float)
    csv_measurements_dict = {
        '키': csv_measurements[0, 0],
        '샅높이': csv_measurements[0, 1],
        '가슴둘레': csv_measurements[0, 2],
        '허리둘레': csv_measurements[0, 3],
        '엉덩이둘레': csv_measurements[0, 4],
        '넙다리둘레': csv_measurements[0, 5],
        '장딴지둘레': csv_measurements[0, 6],
    }

    # 픽셀 거리와 CSV 데이터 병합
    combined_measurements = {**csv_measurements_dict}
    measurements.append(combined_measurements)
    features_list.append(features)
#%%
# 데이터 준비
def pad_features(features_list, max_len):
    """특징 벡터 길이를 최대 길이에 맞춰 패딩"""
    padded_features = []
    for features in features_list:
        if len(features) < max_len:
            # 패딩을 0으로 추가
            padded_features.append(features + [0] * (max_len - len(features)))
        else:
            padded_features.append(features[:max_len])  # 길이가 넘치면 자르기
    return np.array(padded_features)

# 특징 벡터의 최대 길이 구하기
max_feature_len = max(len(features) for features in features_list)

# 특징 벡터 패딩
X = pad_features(features_list, max_feature_len)

# 실제 신체 치수
y = np.array([[m['키'], m['샅높이'], m['가슴둘레'], m['허리둘레'],
               m['엉덩이둘레'], m['넙다리둘레'], m['장딴지둘레']] for m in measurements])

print(X.shape)
print(y.shape)

#%%
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model



# 데이터셋 분할 (훈련 80%, 검증 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 구성
input_layer = layers.Input(shape=(X_train.shape[1],))  # 특징 벡터 크기
x = layers.Dense(1024, activation='relu')(input_layer)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # 30% 드롭아웃
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(7, activation='linear')(x)  # 7개의 출력 (회귀용)

# 최종 모델
model_save_path = r'E:\body\model\trained_transformer_model.keras'

# 저장된 모델 불러오기
model = load_model(model_save_path)

# 모델 구조 확인
model.summary()
#%%
import matplotlib.pyplot as plt

# 모델 훈련
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=1000,
    batch_size=16
)

# 모델 저장
model_save_path = model_save_path.replace('.h5', '.keras')  # 저장 포맷을 .keras로 변경
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# 훈련 과정의 손실과 MAE 추출
history_dict = history.history
loss_values = history_dict['loss']  # 훈련 손실
val_loss_values = history_dict['val_loss']  # 검증 손실
mae_values = history_dict['mae']  # 훈련 MAE
val_mae_values = history_dict['val_mae']  # 검증 MAE

# 그래프 그리기
epochs = range(1, len(loss_values) + 1)

# 손실 그래프
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'b', label='Training loss')
plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# MAE 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, mae_values, 'b', label='Training MAE')
plt.plot(epochs, val_mae_values, 'r', label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

# 그래프 출력
plt.tight_layout()
plt.show()
#%%

