"""
3D ADMM Parameter Sweep Configuration
"""

import os

# 경로 설정
# PSF_DIRECTORY = "/mnt/NAS/Grants/24_AIOBIO/2501_data/calib/mask_1/whole_psf"
PSF_DIRECTORY = "/mnt/NAS/Grants/25_AIOBIO/experiment/250702_psf_measure/"
# RAW_IMAGE_PATH = "../../forward_simulation/convolved_image/sum/simulated_raw_sum_type_1_0701_0825.png"
RAW_IMAGE_PATH = "/mnt/NAS/Grants/25_AIOBIO/experiment/250703_raw_measure/inpainting/07.0.png"
# RAW_IMAGE_PATH = "/home/hotdog/sample/25AIOBIO/forward_simulation/convolved_image/sum/250704_simulated_raw_sum_type_1_0704_0844.png"
# RAW_IMAGE_PATH = "/home/hotdog/sample/25AIOBIO/_ForwardSimulation/convolved_image/sum/250704_simulated_raw_sum_old_type2_0706_1527.png"

RESULT_BASE_PATH = "/mnt/NAS/Grants/25_AIOBIO/experiment/250729_recon/"
# save in NAS

if not os.path.isdir(RESULT_BASE_PATH):
    os.makedirs(RESULT_BASE_PATH)
    print(f'{RESULT_BASE_PATH} directory is created')

# PSF 고정 설정 (30:61:1 패턴)
PSF_PRESET_CONFIG = {
    "start_index": 17,
    "end_index": 39,    # 61 미포함 (즉, 60까지)
    "step_size": 2      # 30, 31, 32, ..., 60 (총 31개)
}

# 기존 그룹 기반 설정 (사용 안함)
# PSF_PRESET_CONFIG_OLD = {
#     "start_index": 25,
#     "step_size": 15,
#     "num_groups": 3
# }

# ADMM 고정 파라미터 (3D 전용, grayscale)
FIXED_CONFIG = {
    # 파일 경로 관련
    'path_ref': 0,
    'save_every': 200,  # Save .mat every N iterations
    'save_fig': True,
    'show_figs': False,  # GUI 없는 환경

    # 데이터 처리 (3D grayscale)
    'color_to_process': 'mono',  # 3D는 grayscale만
    'image_bias': 0,
    'psf_bias': 0,
    'raw_bias': 0,

    # 다운샘플링
    'lateral_downsample': 8,
    'axial_downsample': 1,
    'start_z': 0,  # 첫 번째 plane (1-indexed)
    'end_z': 0,    # 마지막 plane (0이면 자동)

    # GPU 설정
    'useGPU': True,
    'numGPU': 0,

    # 반복 설정
    'max_iter': 1000,
    'disp_figs': 200,  # 200번마다 figure 표시/저장
    'print_interval': 100,
    'regularizer': '3dtvz',  # 기본값 (파라미터로 덮어씀)

    # 자동 튜닝 개선 설정
    'autotune': 1,  # 자동 튜닝 활성화
    'autotune_start_iter': 50,  # 50회 반복 후 자동 튜닝 시작
    'autotune_interval': 10,  # 10회마다 파라미터 조정 체크
    'mu_inc': 1.15,  # 보다 부드러운 증가 (기존 1.2 → 1.15)
    'mu_dec': 1.15,  # 보다 부드러운 감소 (기존 1.2 → 1.15)
    'resid_tol': 1.3,  # 더 엄격한 tolerance (기존 1.5 → 1.3)

    # 적응적 파라미터 조정 설정
    'adaptive_tau': True,  # tau 값 적응적 조정
    'tau_adaptation_rate': 0.95,  # tau 조정 비율
    'convergence_window': 20,  # 수렴 판단 윈도우
    'min_improvement_threshold': 1e-4,  # 최소 개선 임계값

    # 디스플레이 설정
    'roih': 700,
    'roiw': 700,
    'display_norm_method': 'log',
    'beta_z': 10
}

# 파라미터 조합들 (mu1, mu2, mu3, tau, tau_z, tau_n, regularizer)
PARAM_COMBINATIONS = [
    (0.3, 0.68, 3.5, 6e-4, 6e-6, 0.06, 'anisotropic'),
    (0.4, 0.7, 3.0, 6e-4, 6e-5, 0.06, 'anisotropic'),
    (0.6, 0.6, 3.5, 8e-4, 6e-6, 0.08, '3dtv'),
    (0.5, 0.4, 5.2, 6e-4, 6e-5, 0.06, '3dtv'),
    (0.6, 0.68, 4.0, 1e-3, 1e-5, 0.1, 'anisotropic'),
]

# 🎯 스마트 파라미터 범위 기반 자동 생성 설정
SMART_PARAM_RANGES = {
    'mu1': [0.3, 0.4, 0.5, 0.6],  # 데이터 피델리티 가중치
    'mu2': [0.5, 0.68, 0.8],      # 정규화 가중치
    'mu3': [3.0, 3.5, 4.0, 5.0],  # 비음수 제약 가중치
    'tau': [6e-4, 8e-4, 1e-3],    # TV 정규화 강도
    'tau_z': [6e-6, 6e-5, 1e-5],  # Z축 정규화 강도
    'tau_n': [0.06, 0.08, 0.1],   # 네이티브 희소성
    'regularizer': ['anisotropic', '3dtv', '3dtvz']  # 정규화 방법
}

# 자동 조합 생성 모드 ('manual', 'smart_grid', 'adaptive')
AUTO_PARAM_MODE = 'smart_grid'  # 기본값: 수동 조합 사용

# 스마트 그리드 탐색 설정
SMART_GRID_CONFIG = {
    'max_combinations': 15,  # 최대 조합 수 제한
    'priority_weights': {    # 파라미터 중요도 가중치
        'mu1': 0.3,
        'mu2': 0.3,
        'mu3': 0.2,
        'tau': 0.1,
        'tau_z': 0.05,
        'tau_n': 0.05
    },
    'exploration_strategy': 'balanced',  # 'conservative', 'balanced', 'aggressive'
}

# 적응적 파라미터 조정 설정
ADAPTIVE_CONFIG = {
    'initial_params': {
        'mu1': 0.4,
        'mu2': 0.68,
        'mu3': 3.5,
        'tau': 6e-4,
        'tau_z': 6e-6,
        'tau_n': 0.06,
        'regularizer': 'anisotropic'
    },
    'adjustment_rules': {
        'high_noise': {'tau': 1.5, 'tau_z': 1.5},  # 노이즈 높으면 정규화 강화
        'low_contrast': {'mu1': 0.8, 'mu2': 1.2},  # 대비 낮으면 데이터 피델리티 감소
        'over_smoothing': {'tau': 0.7, 'tau_z': 0.7},  # 과도한 스무딩시 정규화 완화
    }
}

# 로깅 설정
LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILENAME = "parameter_sweep_3d.log"


def get_experiment_name(mu1, mu2, mu3, tau, tau_z, tau_n, regularizer):
    return f"mu1_{mu1:.2f}_mu2_{mu2:.2f}_mu3_{mu3:.1f}_tau_{tau:.0e}_tauz_{tau_z:.0e}_taun_{tau_n:.2f}_{regularizer}"


def generate_smart_param_combinations():
    """
    스마트 파라미터 조합 생성 함수

    Returns:
        List[Tuple]: 최적화된 파라미터 조합 리스트
    """
    import itertools
    import random

    if AUTO_PARAM_MODE == 'manual':
        return PARAM_COMBINATIONS

    elif AUTO_PARAM_MODE == 'smart_grid':
        # 스마트 그리드 탐색
        combinations = []

        # 핵심 파라미터 조합 먼저 생성
        core_combinations = list(itertools.product(
            SMART_PARAM_RANGES['mu1'][:2],  # 상위 2개만
            SMART_PARAM_RANGES['mu2'][:2],  # 상위 2개만
            SMART_PARAM_RANGES['mu3'][:2],  # 상위 2개만
            SMART_PARAM_RANGES['tau'][:2],  # 상위 2개만
            SMART_PARAM_RANGES['tau_z'][:2],  # 상위 2개만
            SMART_PARAM_RANGES['tau_n'][:2],  # 상위 2개만
            SMART_PARAM_RANGES['regularizer'][:2]  # 상위 2개만
        ))

        # 전체 조합에서 무작위 선택
        all_combinations = list(
            itertools.product(*SMART_PARAM_RANGES.values()))

        # 핵심 조합 + 추가 무작위 조합
        max_combinations = SMART_GRID_CONFIG['max_combinations']

        if len(core_combinations) < max_combinations:
            remaining_slots = max_combinations - len(core_combinations)
            additional_combinations = random.sample(
                [combo for combo in all_combinations if combo not in core_combinations],
                min(remaining_slots, len(all_combinations) - len(core_combinations))
            )
            combinations = core_combinations + additional_combinations
        else:
            combinations = core_combinations[:max_combinations]

        print(f"스마트 그리드 탐색: {len(combinations)}개 조합 생성")
        return combinations

    elif AUTO_PARAM_MODE == 'adaptive':
        # 적응적 조정은 단일 초기 파라미터에서 시작
        initial = ADAPTIVE_CONFIG['initial_params']
        return [(
            initial['mu1'],
            initial['mu2'],
            initial['mu3'],
            initial['tau'],
            initial['tau_z'],
            initial['tau_n'],
            initial['regularizer']
        )]

    else:
        # 기본값: 수동 조합 반환
        return PARAM_COMBINATIONS


def get_data_adaptive_params(raw_image_path=None):
    """
    데이터 특성에 따른 적응적 파라미터 조정

    Args:
        raw_image_path (str): raw 이미지 경로 (분석용)

    Returns:
        Dict: 조정된 파라미터
    """
    import cv2  # type: ignore
    import numpy as np

    base_params = ADAPTIVE_CONFIG['initial_params'].copy()

    if raw_image_path and os.path.exists(raw_image_path):
        try:
            # 이미지 로드 및 분석
            img = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 노이즈 레벨 추정 (Laplacian variance 방법)
                noise_level = cv2.Laplacian(img, cv2.CV_64F).var()

                # 대비 분석
                contrast = img.std()

                # 조정 규칙 적용
                if noise_level > 1000:  # 높은 노이즈
                    adjustments = ADAPTIVE_CONFIG['adjustment_rules']['high_noise']
                    for param, factor in adjustments.items():
                        if param in base_params:
                            base_params[param] *= factor
                    print(f"높은 노이즈 감지 (레벨: {noise_level:.0f}): 정규화 강화")

                if contrast < 30:  # 낮은 대비
                    adjustments = ADAPTIVE_CONFIG['adjustment_rules']['low_contrast']
                    for param, factor in adjustments.items():
                        if param in base_params:
                            base_params[param] *= factor
                    print(f"낮은 대비 감지 (대비: {contrast:.1f}): 데이터 피델리티 조정")

        except Exception as e:
            print(f"이미지 분석 중 오류: {e}")

    return base_params
