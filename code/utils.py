import numpy as np
import os
import cv2
from tqdm import tqdm


def debayer_RGGB_G(img_path, normalize=True):
    """
    Load raw .npy, extract Green channels, average them, and optionally normalize.
    """
    # 1. Load numpy
    try:
        raw_data_u8 = np.load(img_path)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

    # 2. 8-bit * 2 -> 16-bit view
    # 입력 데이터가 uint8로 패킹되어 있다고 가정
    raw_data_16 = raw_data_u8.view(np.uint16)

    # 3. Split the channels (Green extraction)
    # Green1: 짝수행, 홀수열 (0,1), (0,3)...
    # Green2: 홀수행, 짝수열 (1,0), (1,2)...
    raw_green1 = raw_data_16[0::2, 1::2]
    raw_green2 = raw_data_16[1::2, 0::2]

    # 4. Green Average (float32 변환 필수)
    # 문법 수정: raw_green1(np.float32) -> raw_green1.astype(np.float32)
    raw_green = (raw_green1.astype(np.float32) + raw_green2.astype(np.float32)) / 2.0

    # 5. [중요] Normalize (0 ~ 1)
    if normalize:
        max_val = raw_green.max()
        if max_val > 0:
            raw_green = raw_green / max_val

    return raw_green


def background_remove(img_dir, background_path, save_dir):
    """
    img_dir: 이미지 npy 파일들이 있는 폴더
    background_path: background npy 파일 경로
    save_dir: 결과 저장할 폴더
    """

    # background 불러오기
    bg_data_u8 = np.load(background_path)

    # 저장 폴더 없으면 생성
    os.makedirs(save_dir, exist_ok=True)

    # 폴더 내 모든 npy 파일 처리
    file_list = [f for f in os.listdir(img_dir) if f.endswith(".npy")]

    for fname in tqdm(file_list, desc="Background removing"):
        # 원본 불러오기
        raw_data_u8 = np.load(os.path.join(img_dir, fname))
        # raw_data_16 = raw_data_u8.view(np.uint16)

        # background 제거
        # result = raw_data_16.astype(np.int32) - bg_data_16.astype(np.int32)
        result = raw_data_u8.astype(np.int32) - bg_data_u8.astype(np.int32)
        result = np.clip(result, 0, None).astype(np.uint8)

        # 저장
        save_path = os.path.join(save_dir, fname.replace(".npy", "_bg_removed.npy"))
        np.save(save_path, result)
