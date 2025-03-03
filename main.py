import os, sys
import cv2
from PIL import Image
import numpy as np
import glob
import warnings
import argparse
from cloths_segmentation.pre_trained_models import create_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()
    img = cv2.imread("./static/origin_web.jpg")
    if img is None:
        print("이미지를 읽지 못했습니다. 경로를 확인하세요.")
    else:
        print("원본 이미지 크기:", img.shape)

        # 목표 크기
        target_width = 768
        target_height = 1024

        # 원본 이미지 크기 가져오기
        h, w = img.shape[:2]

        # 비율 계산
        scale = min(target_width / w, target_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)

        # 원본 이미지를 비율에 맞게 리사이즈
        resized_img = cv2.resize(img, (new_width, new_height))

        # 검은색 캔버스 생성 (목표 크기)
        canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

        # 중앙에 이미지 배치
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

        # 저장
        result = cv2.imwrite("./static/origin_web.jpg", canvas)
        if not result:
            print("이미지 저장에 실패했습니다. 경로와 권한을 확인하세요.")
        else:
            print("이미지 저장 성공: 검은색 여백으로 확장 완료")
    # Read input image
    img=cv2.imread("./static/origin_web.jpg")
    ori_img=cv2.resize(img,(768,1024))
    cv2.imwrite("./origin.jpg",ori_img)
    print('성공1')
    # Resize input image
    img=cv2.imread('origin.jpg')
    img=cv2.resize(img,(384,512))
    cv2.imwrite('resized_img.jpg',img)
    print('성공2')
    # Get mask of cloth
    print("Get mask of cloth\n")
    terminnal_command = "python get_cloth_mask.py"
    os.system(terminnal_command)
    print('성공3')
    # Get openpose coordinate using posenet
    print("Get openpose coordinate using posenet\n")
    terminnal_command = "python posenet.py"
    os.system(terminnal_command)
    print('성공4')
    # Generate semantic segmentation using Graphonomy-Master library
    print("Generate semantic segmentation using Graphonomy-Master library\n")
    os.chdir("./Graphonomy-master")
    print('성공44')
    terminnal_command ="python exp/inference/inference.py --loadmodel ./inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img"
    os.system(terminnal_command)
    os.chdir("../")
    print('성공5')
    # Remove background image using semantic segmentation mask
    mask_img=cv2.imread('./resized_segmentation_img.png',cv2.IMREAD_GRAYSCALE)
    mask_img=cv2.resize(mask_img,(768,1024))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_img = cv2.erode(mask_img, k)
    img_seg=cv2.bitwise_and(ori_img,ori_img,mask=mask_img)
    back_ground=ori_img-img_seg
    img_seg=np.where(img_seg==0,215,img_seg)
    cv2.imwrite("./seg_img.png",img_seg)
    img=cv2.resize(img_seg,(768,1024))
    cv2.imwrite('./HR-VITON-main/test/test/image/00001_00.jpg',img)
    print('성공6')
    # Generate grayscale semantic segmentation image
    terminnal_command ="python get_seg_grayscale.py"
    os.system(terminnal_command)
    print('성공7')
    # Generate Densepose image using detectron2 library
    print("\nGenerate Densepose image using detectron2 library\n")
    terminnal_command ="python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    origin.jpg --output output.pkl -v"
    os.system(terminnal_command)
    terminnal_command ="python get_densepose.py"
    os.system(terminnal_command)

    # Run HR-VITON to generate final image
    print("\nRun HR-VITON to generate final image\n")
    os.chdir("./HR-VITON-main")
    print('1')
    terminnal_command = "python3 test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 --gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test"
    print('2')
    os.system(terminnal_command)
    print('3')
##확인
    terminnal_command = (
        "python test_generator.py --cuda True --test_name test1 --tocg_checkpoint mtviton.pth --gpu_ids 0 "
        "--gen_checkpoint gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test > test_generator_log.txt 2>&1"
    )
    exit_code = os.system(terminnal_command)
    print(f"HR-VITON test_generator.py 실행 결과: {exit_code}")
    print("로그는 test_generator_log.txt 파일에서 확인하세요.")
##
    # Add Background or Not
    l=glob.glob("./Output/*.png")

    # Add Background
    if opt.background:
        for i in l:
            img=cv2.imread(i)
            img=cv2.bitwise_and(img,img,mask=mask_img)
            img=img+back_ground
            cv2.imwrite(i,img)

    # Remove Background
    else:
        for i in l:
            img=cv2.imread(i)
            cv2.imwrite(i,img)

    os.chdir("../")
    cv2.imwrite("./static/finalimg.png", img)