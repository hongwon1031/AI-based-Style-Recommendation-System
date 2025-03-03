import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

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

# 데이터 증강 설정
datagen = ImageDataGenerator(
    width_shift_range=0.1,     # 가로 이동 범위
    height_shift_range=0.1,    # 세로 이동 범위
    shear_range=0.1,           # 전단 변환 범위
    zoom_range=0.1,            # 확대/축소 범위
    horizontal_flip=False,      # 좌우 대칭
    fill_mode='nearest'        # 빈 픽셀 보간
)
#%%
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
#%%

#image_dir = r'E:\body\densepose_images'
#csv_dir = r'E:\body\csvfiles'
image_dir = r'E:\body\densepose_images'
csv_dir = r'E:\body\csvfiles'
# 이미지 및 측정값 저장 리스트
features_list = []
measurements = []

# 이미지와 CSV 파일 이름 정렬
image_names = sorted(os.listdir(image_dir))
csv_names = sorted(os.listdir(csv_dir))
#%%
# 이미지와 CSV 매칭 및 처리
for image_name in image_names:
    # 이미지 ID 추출
    image_identifier = os.path.splitext(image_name)[0]  # 확장자 제거
    print(f"Processing image: {image_name} (ID: {image_identifier})")  # 디버깅 로그
    # 해당하는 CSV 찾기
    matched_csv = next((csv_name for csv_name in csv_names if image_identifier in csv_name), None)
    if not matched_csv:
        continue  # 매칭 실패 시 건너뜀

    # 이미지 로드
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)
    image_data = np.array(image)
    print(f"Loaded image: {image_name}, Shape: {image_data.shape}")  # 디버깅 로그
    # 데이터 증강 적용
    augmented_images = datagen.flow(np.expand_dims(image_data, axis=0), batch_size=1)
    for _ in range(5):  # 각 이미지당 5개의 증강 데이터 생성
        augmented_image = next(augmented_images)[0].astype('uint8')
        # 신체 부위별 특징 추출
        features = extract_body_part_features(augmented_image)
        features_list.append(features)

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

# 결과 확인
print(f"총 생성된 특징 벡터 개수: {len(features_list)}")
print(f"총 측정값 개수: {len(measurements)}")



#%%
print(len(features_list[0]))
print(measurements[0])
#%%
# 데이터 준비
max_feature_len = max(len(features) for features in features_list)

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

# 특징 벡터 패딩
X = pad_features(features_list, max_feature_len)

# 실제 신체 치수
y = np.array([[m['키'], m['샅높이'], m['가슴둘레'], m['허리둘레'],
               m['엉덩이둘레'], m['넙다리둘레'], m['장딴지둘레']] for m in measurements])

print(X.shape)
print(y.shape)

#%%


# 키와 샅높이를 저장할 리스트
X_specific = []


def calculate_height_and_crotch_height(image_data):
    """
    이미지에서 키와 샅높이를 계산
    - image_data: 세그멘테이션 이미지 데이터 (H, W, C)
    - return: 키(pixel_height), 샅높이(crotch_height)
    """
    # RGB 색상 정의
    head_color = (255, 0, 0)  # 머리
    left_leg_color = (85, 255, 170)  # 왼쪽 다리(종아리 포함)
    torso_color = (85, 51, 0)  # 몸통

    # 머리 마스크 생성
    head_mask = np.all(image_data == head_color, axis=-1)
    head_y_coords = np.where(head_mask)[0]  # Y좌표 추출

    # 왼쪽 다리 마스크 생성
    left_leg_mask = np.all(image_data == left_leg_color, axis=-1)
    left_leg_y_coords = np.where(left_leg_mask)[0]  # Y좌표 추출

    # 몸통 마스크 생성
    torso_mask = np.all(image_data == torso_color, axis=-1)
    torso_y_coords = np.where(torso_mask)[0]  # Y좌표 추출

    # 결과 초기화
    pixel_height = None
    crotch_height = None

    # 키 계산
    if len(head_y_coords) > 0 and len(left_leg_y_coords) > 0:
        head_min_y = np.min(head_y_coords)
        left_leg_max_y = np.max(left_leg_y_coords)
        pixel_height = left_leg_max_y - head_min_y

    # 샅높이 계산
    if len(torso_y_coords) > 0 and len(left_leg_y_coords) > 0:
        torso_min_y = np.min(torso_y_coords)
        left_leg_max_y = np.max(left_leg_y_coords)
        crotch_height = left_leg_max_y - torso_min_y

    return pixel_height, crotch_height



# 모든 이미지에 대해 키와 샅높이를 계산
for image_name in image_names:
    # 이미지 경로 생성
    image_path = os.path.join(image_dir, image_name)

    # 이미지 로드
    image = Image.open(image_path)
    image_data = np.array(image)

    # 키와 샅높이 계산
    height, crotch_height = calculate_height_and_crotch_height(image_data)

    # 키와 샅높이를 X_specific에 추가
    if height is not None and crotch_height is not None:
        X_specific.append([height, crotch_height])  # 두 값을 리스트로 추가
    else:
        X_specific.append([None, None])  # 계산이 실패한 경우 None으로 추가

    # 디버깅 로그 출력
    print(f"Processed {image_name}: Height = {height}, Crotch Height = {crotch_height}")

# 결과 확인
print(f"총 계산된 키와 샅높이 데이터 개수: {len(X_specific)}")
print(X_specific[0])
print(X[0])
print(y[0])
print(len(X))
print(len(y))
#%%
#####################################################################건들지마################################
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=40)  # 차원을 40으로 축소
X_reduced = pca.fit_transform(X_scaled)
print(f"Reduced feature shape: {X_reduced.shape}")

output_scaler = StandardScaler()
y_scaled = output_scaler.fit_transform(y)

#%%
# 데이터셋 분할 (훈련 80%, 검증 20%)
X_train, X_val, y_train, y_val = train_test_split(X_reduced, y_scaled, test_size=0.2, random_state=42)

#%%
print(X_train[0])
print(y_train[0])
print(np.isnan(X_train).any(), np.isinf(X_train).any())
print(np.isnan(y_train).any(), np.isinf(y_train).any())
# y_train에서 NaN 값의 위치 확인
nan_indices = np.isnan(y_train)
print(f"NaN 값의 인덱스: {np.where(nan_indices)}")
# NaN 값이 있는 행을 제거
valid_indices = ~np.isnan(y_train).any(axis=1)
X_train = X_train[valid_indices]
y_train = y_train[valid_indices]
print(X_train[0])
print(y_train[0])
print(np.isnan(X_train).any(), np.isinf(X_train).any())
print(np.isnan(y_train).any(), np.isinf(y_train).any())
#%%



from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import MeanAbsoluteError

# 기존 모델 로드
model_save_path = r'E:\body\model\my_model_2.h5'
model = load_model(model_save_path)
optimizer = Adam(learning_rate=0.0001)
losses = {
    'output_height': 'mae',
    'output_crotch_height': 'mae',
    'output_chest_girth': 'mae',
    'output_waist_girth': 'mae',
    'output_hip_girth': 'mae',
    'output_thigh_girth': 'mae',
    'output_calf_girth': 'mae',
}
metrics = {
    'output_height': [MeanAbsoluteError()],
    'output_crotch_height': [MeanAbsoluteError()],
    'output_chest_girth': [MeanAbsoluteError()],
    'output_waist_girth': [MeanAbsoluteError()],
    'output_hip_girth': [MeanAbsoluteError()],
    'output_thigh_girth': [MeanAbsoluteError()],
    'output_calf_girth': [MeanAbsoluteError()],
}
# 중간 레이어 가져오기 (dense_38까지)
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)

# 멀티-헤드 출력층 추가
x = feature_extractor.output
output_1 = Dense(1, name='output_height')(x)  # 키
output_2 = Dense(1, name='output_crotch_height')(x)  # 샅높이
output_3 = Dense(1, name='output_chest_girth')(x)  # 가슴둘레
output_4 = Dense(1, name='output_waist_girth')(x)  # 허리둘레
output_5 = Dense(1, name='output_hip_girth')(x)  # 엉덩이둘레
output_6 = Dense(1, name='output_thigh_girth')(x)  # 넙다리둘레
output_7 = Dense(1, name='output_calf_girth')(x)  # 장딴지둘레
# 새 모델 생성
model = Model(inputs=feature_extractor.input, outputs=[output_1, output_2, output_3, output_4, output_5, output_6, output_7])

# 기존 레이어 동결 (가중치 고정)
for layer in feature_extractor.layers[-5:]:  # 마지막 5개 레이어만 학습 가능
    layer.trainable = True

model.summary()
#%%

# 체크포인트 저장 경로
checkpoint_dir = r'E:\body\model'
os.makedirs(checkpoint_dir, exist_ok=True)

# ModelCheckpoint 콜백 설정
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'my_model_3_each_epoch_{epoch:03d}.keras'),  # epoch 번호 포함
    save_freq=100 * len(X_train) // 16,  # 400 epoch마다 저장, len(X_train)/batch_size로 계산
    save_best_only=False  # 매번 새로운 모델 저장
)

# 컴파일 및 학습
model.compile(optimizer='adam', loss=losses, metrics=metrics)
history = model.fit(X_train, {
    'output_height': y_train[:, 0],
    'output_crotch_height': y_train[:, 1],
    'output_chest_girth': y_train[:, 2],
    'output_waist_girth': y_train[:, 3],
    'output_hip_girth': y_train[:, 4],
    'output_thigh_girth': y_train[:, 5],
    'output_calf_girth': y_train[:, 6],
}, epochs=1000, batch_size=32,
                    callbacks=[checkpoint_callback])  # 콜백 추가


#%%
new_model_save_path = r'E:\body\model\my_model_4_each.h5'
# 모델 저장
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(new_model_save_path)
print(f"Model saved at: {new_model_save_path}")
history_dict = history.history
print(history_dict.keys())

#%%
# 출력값별 MAE 키 추출

mae_keys = [key for key in history_dict.keys() if 'mean_absolute_error' in key and not key.startswith('val_')]
val_mae_keys = [key for key in history_dict.keys() if key.startswith('val_') and 'mean_absolute_error' in key]

# 디버깅: 키 확인
print("MAE keys:", mae_keys)
print("Validation MAE keys:", val_mae_keys)

import numpy as np

# 정규화 해제 함수
def restore_mae(mae_values, scaler, feature_idx):
    """
    정규화된 MAE를 원래 단위로 복원합니다.
    - mae_values: 정규화된 MAE 값 (list 또는 array)
    - scaler: StandardScaler 객체
    - feature_idx: 복원할 출력값의 인덱스 (0부터 시작)
    """
    scale = scaler.scale_[feature_idx]
    return np.array(mae_values) * scale

# 정규화된 MAE 복원 및 출력
restored_mae = {}
for key in mae_keys:
    output_name = key.replace('_mean_absolute_error', '')  # 'output_calf_girth' 등 추출
    feature_idx = mae_keys.index(key)  # 해당 출력값의 인덱스
    restored_mae[output_name] = restore_mae(history_dict[key], output_scaler, feature_idx)

# 에포크 범위 설정
epochs = range(1, len(history_dict[mae_keys[0]]) + 1)

# 그래프 그리기
plt.figure(figsize=(12, 8))

# 각 출력값별 MAE 그래프 (정규화된 MAE)
for key in mae_keys:
    plt.plot(epochs, history_dict[key], label=f'Training {key} (Normalized)')
for key in val_mae_keys:
    plt.plot(epochs, history_dict[key], linestyle='dashed', label=f'Validation {key} (Normalized)')

# 복원된 MAE 그래프 (원래 단위)
for output_name, values in restored_mae.items():
    plt.plot(epochs, values, linestyle='dotted', label=f'Training {output_name} (Restored)')

# 그래프 설정
plt.title('MAE for Each Output (Normalized and Restored)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.tight_layout()

# 그래프 출력
plt.show()

# 복원된 MAE 값 출력 (마지막 에포크 기준)
print("\nRestored MAE (Original Scale):")
for output_name, values in restored_mae.items():
    print(f"{output_name}: {values[-1]:.2f}")

#%%
##############################################################새로 학습##################################################
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
import numpy as np


print(history.history.keys())
#%%
early_stopping = EarlyStopping(
    monitor='loss',  # 특정 출력 손실을 모니터링
    patience=20,
    restore_best_weights=True,
    verbose=1,
    mode='min'
)


lr_scheduler = ReduceLROnPlateau(
    monitor='loss',  # 특정 출력 손실을 모니터링
    factor=0.5,
    patience=100,
    min_lr=1e-6,
    verbose=1,
    mode='min'
)
checkpoint_dir = r'E:\body\model'
os.makedirs(checkpoint_dir, exist_ok=True)

# ModelCheckpoint 콜백 설정
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'my_model_6_each_epoch_{epoch:03d}.keras'),  # epoch 번호 포함
    save_freq=100 * len(X_train) // 16,  # 400 epoch마다 저장, len(X_train)/batch_size로 계산
    save_best_only=False  # 매번 새로운 모델 저장
)



# 기존 모델 로드
model_save_path = r'E:\body\model\my_model_5_each.h5'
model = load_model(model_save_path, compile=False)  # 기존 모델 구조와 가중치 로드
###################################3추가###########################
x = model.get_layer('dense_38').output

# 추가 레이어 구성
x = Dense(256, activation='relu', name='added_dense_1')(x)  # 추가된 첫 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_1')(x)  # Dropout 레이어 추가
x = Dense(256, activation='relu', name='added_dense_2')(x)  # 추가된 두 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_2')(x)  # Dropout 레이어 추가
x = Dense(256, activation='relu', name='added_dense_3')(x)  # 추가된 첫 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_3')(x)  # Dropout 레이어 추가
x = Dense(128, activation='relu', name='added_dense_4')(x)  # 추가된 두 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_4')(x)  # Dropout 레이어 추가
x = Dense(128, activation='relu', name='added_dense_5')(x)  # 추가된 첫 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_5')(x)  # Dropout 레이어 추가
x = Dense(128, activation='relu', name='added_dense_6')(x)  # 추가된 두 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_6')(x)  # Dropout 레이어 추가
x = Dense(64, activation='relu', name='added_dense_7')(x)  # 추가된 첫 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_7')(x)  # Dropout 레이어 추가
x = Dense(64, activation='relu', name='added_dense_8')(x)  # 추가된 두 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_8')(x)  # Dropout 레이어 추가
x = Dense(64, activation='relu', name='added_dense_9')(x)  # 추가된 첫 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_9')(x)  # Dropout 레이어 추가
x = Dense(32, activation='relu', name='added_dense_10')(x)  # 추가된 두 번째 Dense 레이어
x = layers.Dropout(0.3, name='added_dropout_10')(x)  # Dropout 레이어 추가
x = Dense(32, activation='relu', name='added_dense_11')(x)  # 추가된 세 번째 Dense 레이어

# 각 출력층에 연결
output_1 = Dense(1, name='output_height')(x)
output_2 = Dense(1, name='output_crotch_height')(x)
output_3 = Dense(1, name='output_chest_girth')(x)
output_4 = Dense(1, name='output_waist_girth')(x)
output_5 = Dense(1, name='output_hip_girth')(x)
output_6 = Dense(1, name='output_thigh_girth')(x)
output_7 = Dense(1, name='output_calf_girth')(x)

# 새로운 모델 정의
new_model = Model(inputs=model.input, outputs=[output_1, output_2, output_3, output_4, output_5, output_6, output_7])

# 기존 레이어 가중치 유지 (필요시 동결 가능)
for layer in new_model.layers[:-18]:  # 마지막 10개의 레이어는 새로 추가된 것
    layer.trainable = True  # 기존 레이어를 학습 가능 상태로 유지
    # 필요에 따라 layer.trainable = False로 설정하여 동결
###############################3추가####################################


optimizer = Adam(learning_rate=0.00001)
losses = {
    'output_height': 'mae',
    'output_crotch_height': 'mae',
    'output_chest_girth': 'mae',
    'output_waist_girth': 'mae',
    'output_hip_girth': 'mae',
    'output_thigh_girth': 'mae',
    'output_calf_girth': 'mae',
}
loss_weights = {
    'output_height': 2.0,  # 높게 설정하여 더 집중적으로 학습
    'output_crotch_height': 1.0,
    'output_chest_girth': 1.0,
    'output_waist_girth': 1.0,
    'output_hip_girth': 1.0,
    'output_thigh_girth': 1.0,
    'output_calf_girth': 1.0,
}
metrics = {
    'output_height': [MeanAbsoluteError()],
    'output_crotch_height': [MeanAbsoluteError()],
    'output_chest_girth': [MeanAbsoluteError()],
    'output_waist_girth': [MeanAbsoluteError()],
    'output_hip_girth': [MeanAbsoluteError()],
    'output_thigh_girth': [MeanAbsoluteError()],
    'output_calf_girth': [MeanAbsoluteError()],
}
###수정됨###
'''for layer in model.layers:
    layer.trainable = True  # 필요한 경우, 특정 레이어만 True로 변경 가능'''
model.summary()
###수정됨


new_model.summary()
#%%

# 체크포인트 저장 경로

#####################################################수정##########################################
'''
# 컴파일 및 학습
model.compile(optimizer=optimizer, loss=losses,loss_weights=loss_weights, metrics=metrics)
history = model.fit(X_train, {
    'output_height': y_train[:, 0],
    'output_crotch_height': y_train[:, 1],
    'output_chest_girth': y_train[:, 2],
    'output_waist_girth': y_train[:, 3],
    'output_hip_girth': y_train[:, 4],
    'output_thigh_girth': y_train[:, 5],
    'output_calf_girth': y_train[:, 6],
}, epochs=300, batch_size=16,
                    callbacks=[lr_scheduler, checkpoint_callback, early_stopping])  # 콜백 추가'''
#####################################################수정##########################################
new_model.compile(optimizer=optimizer, loss=losses,loss_weights=loss_weights, metrics=metrics)
history = new_model.fit(X_train, {
    'output_height': y_train[:, 0],
    'output_crotch_height': y_train[:, 1],
    'output_chest_girth': y_train[:, 2],
    'output_waist_girth': y_train[:, 3],
    'output_hip_girth': y_train[:, 4],
    'output_thigh_girth': y_train[:, 5],
    'output_calf_girth': y_train[:, 6],
}, epochs=300, batch_size=16,
                    callbacks=[lr_scheduler, checkpoint_callback, early_stopping])  # 콜백 추가
#%%

history_dict = history.history
print(history_dict.keys())
# 출력값별 MAE 키 추출

mae_keys = [key for key in history_dict.keys() if 'mean_absolute_error' in key and not key.startswith('val_')]
val_mae_keys = [key for key in history_dict.keys() if key.startswith('val_') and 'mean_absolute_error' in key]

# 디버깅: 키 확인
print("MAE keys:", mae_keys)
print("Validation MAE keys:", val_mae_keys)
# 복원된 MAE 그래프 (원래 단위)
for output_name, values in restored_mae.items():
    plt.plot(epochs, values, linestyle='dotted', label=f'Training {output_name} (Restored)')

# 복원된 MAE 값 출력 (마지막 에포크 기준)
print("\nRestored MAE (Original Scale):")
for output_name, values in restored_mae.items():
    print(f"{output_name}: {values[-1]:.2f}")


#%%



# 정규화 해제 함수
def restore_mae(mae_values, scaler, feature_idx):
    """
    정규화된 MAE를 원래 단위로 복원합니다.
    - mae_values: 정규화된 MAE 값 (list 또는 array)
    - scaler: StandardScaler 객체
    - feature_idx: 복원할 출력값의 인덱스 (0부터 시작)
    """
    scale = scaler.scale_[feature_idx]
    return np.array(mae_values) * scale

# 정규화된 MAE 복원 및 출력
restored_mae = {}
for key in mae_keys:
    output_name = key.replace('_mean_absolute_error', '')  # 'output_calf_girth' 등 추출
    feature_idx = mae_keys.index(key)  # 해당 출력값의 인덱스
    restored_mae[output_name] = restore_mae(history_dict[key], output_scaler, feature_idx)

# 에포크 범위 설정
epochs = range(1, len(history_dict[mae_keys[0]]) + 1)

# 그래프 그리기
plt.figure(figsize=(12, 8))

# 각 출력값별 MAE 그래프 (정규화된 MAE)
for key in mae_keys:
    plt.plot(epochs, history_dict[key], label=f'Training {key} (Normalized)')
for key in val_mae_keys:
    plt.plot(epochs, history_dict[key], linestyle='dashed', label=f'Validation {key} (Normalized)')



# 그래프 설정
plt.title('MAE for Each Output (Normalized and Restored)')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True)
plt.tight_layout()

# 그래프 출력
plt.show()


new_model_save_path = r'E:\body\model\my_model_6_each.h5'
# 모델 저장
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(new_model_save_path)
print(f"Model saved at: {new_model_save_path}")
#%%

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from PIL import Image

# 모델 경로 및 StandardScaler 로드
model_save_path = r'E:\body\model\my_model_6_each.h5'
output_scaler = StandardScaler()  # 동일한 스케일러를 로드해야 함
output_scaler.fit(y)  # 학습 시 사용한 y 데이터로 스케일러 적합 (필수)

# 학습한 모델 로드
model = load_model(model_save_path)
model.summary()
#%%
# 정규화 해제 함수
def restore_output(normalized_values, scaler):
    """
    정규화된 값을 원래 단위로 복원
    - normalized_values: 모델 출력값 (정규화된 값, 각 출력별로 분리된 리스트)
    - scaler: 학습 시 사용한 StandardScaler
    """
    normalized_values = np.array(normalized_values).reshape(1, -1)  # 2차원 배열로 변환
    return scaler.inverse_transform(normalized_values)[0]  # 1차원 배열로 복원

# 추론용 이미지 처리 함수
def preprocess_image(image_path, datagen, max_feature_len, pca_model, scaler):
    """
    이미지를 전처리하고 모델 입력 형식에 맞게 변환
    """
    # 이미지 로드 및 변환
    image = Image.open(image_path)
    image_data = np.array(image)

    # 증강 (단일 이미지에서 증강하지 않고, 원본 그대로 사용)
    augmented_images = datagen.flow(np.expand_dims(image_data, axis=0), batch_size=1)
    augmented_image = next(augmented_images)[0].astype('uint8')

    # 특징 추출
    features = extract_body_part_features(augmented_image)

    # 패딩 처리
    if len(features) < max_feature_len:
        features.extend([0] * (max_feature_len - len(features)))
    elif len(features) > max_feature_len:
        features = features[:max_feature_len]

    # PCA 적용
    features_scaled = scaler.transform([features])  # StandardScaler로 정규화
    features_reduced = pca_model.transform(features_scaled)  # PCA 적용

    return features_reduced

# 추론 실행 함수
def predict(image_path):
    """
    주어진 이미지에 대해 신체 측정값을 예측하고 복원
    """
    # 전처리
    features_reduced = preprocess_image(image_path, datagen, max_feature_len, pca, scaler)

    # 모델 예측
    normalized_outputs = model.predict(features_reduced)

    # 다중 출력값 병합 (모든 출력 헤드를 1차원 리스트로 합침)
    normalized_outputs_flat = np.concatenate([output.flatten() for output in normalized_outputs])

    # 정규화 해제 및 결과 복원
    restored_outputs = restore_output(normalized_outputs_flat, output_scaler)

    return restored_outputs


# 예시 이미지 경로
test_image_path = r'E:\body\2.Validation\test_img\01_00_F001_03_segmentation.png'

# 추론 수행
predicted_outputs = predict(test_image_path)

# 결과 출력
output_labels = ['키', '샅높이', '가슴둘레', '허리둘레', '엉덩이둘레', '넙다리둘레', '장딴지둘레']
for label, value in zip(output_labels, predicted_outputs):
    print(f"{label}: {value:.2f}")




test_image_path = r'E:\body\2.Validation\test_img\01_00_F002_03_segmentation.png'

# 추론 수행
predicted_outputs = predict(test_image_path)

# 결과 출력
output_labels = ['키', '샅높이', '가슴둘레', '허리둘레', '엉덩이둘레', '넙다리둘레', '장딴지둘레']
for label, value in zip(output_labels, predicted_outputs):
    print(f"{label}: {value:.2f}")

test_image_path = r'E:\body\2.Validation\test_img\01_00_F003_03_segmentation.png'

# 추론 수행
predicted_outputs = predict(test_image_path)

# 결과 출력
output_labels = ['키', '샅높이', '가슴둘레', '허리둘레', '엉덩이둘레', '넙다리둘레', '장딴지둘레']
for label, value in zip(output_labels, predicted_outputs):
    print(f"{label}: {value:.2f}")

test_image_path = r'E:\body\2.Validation\test_img\01_00_F003_03_segmentation.png'

# 추론 수행
predicted_outputs = predict(test_image_path)

# 결과 출력
output_labels = ['키', '샅높이', '가슴둘레', '허리둘레', '엉덩이둘레', '넙다리둘레', '장딴지둘레']
for label, value in zip(output_labels, predicted_outputs):
    print(f"{label}: {value:.2f}")