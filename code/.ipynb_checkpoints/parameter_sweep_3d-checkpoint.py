#!/usr/bin/env python3
"""
3D ADMM Parameter Sweep Script
==============================

3D reconstruction을 위한 파라미터 스위핑 스크립트입니다.
PSF stack은 고정하고 ADMM 파라미터들만 변경하여 실험합니다.

Usage:
    python parameter_sweep_3d.py
"""

import os
import sys
import time
import logging
from datetime import datetime
import numpy as np

# 3D reconstruction 모듈들 import
try:
    from psf_preprocessing import PSFPreprocessor
    from admm_3d_refactored import ADMM3D
    from regularizers import TV3DRegularizer, L1Regularizer, CenterWeightedRegularizer, AnisotropicDiffusionRegularizer
    import config_3d
except ImportError as e:
    print(f"Import Error: {e}")
    print("필요한 모듈들이 같은 디렉토리에 있는지 확인하세요.")
    sys.exit(1)

def setup_logging():
    """로깅 설정"""
    log_level = getattr(logging, config_3d.LOG_LEVEL.upper())
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    if config_3d.LOG_TO_FILE:
        file_handler = logging.FileHandler(config_3d.LOG_FILENAME)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def prepare_fixed_psf(logger):
    """고정된 PSF stack을 준비합니다."""
    logger.info("PSF stack 준비 중...")
    
    # PSF 전처리기 초기화
    preprocessor = PSFPreprocessor(
        psf_directory=config_3d.PSF_DIRECTORY,
        dummy_strategy="symmetric_padding",
        dummy_layers_between=0,
        dummy_layers_boundary=0
    )
    
    # PSF 파일 발견
    psf_files = preprocessor.discover_psf_files()
    logger.info(f"발견된 PSF 파일 수: {len(psf_files)}")
    
    # 고정된 PSF 인덱스 선택 (범위 기반: 30:61:1)
    config = config_3d.PSF_PRESET_CONFIG
    start = config["start_index"]
    end = config["end_index"]
    step = config["step_size"]
    
    # 연속 범위 선택: range(start, end, step)
    selected_indices = list(range(start, min(end, len(psf_files)), step))
    
    logger.info(f"선택된 PSF 인덱스: {selected_indices}")
    logger.info(f"총 {len(selected_indices)}개 PSF 선택됨")
    
    # PSF stack 처리 (캐싱 사용)
    psf_stack_path, labels = preprocessor.process_psf_stack(
        selected_indices=selected_indices,
        use_cache=True,
        force_rebuild=False
    )
    
    logger.info(f"PSF stack 준비 완료: {psf_stack_path}")
    return psf_stack_path, labels

def create_regularizer(reg_type, device, tau, tau_z, tau_n):
    """정규화기 생성"""
    if reg_type == "3dtv":
        return TV3DRegularizer(device=device, tau=tau, tau_z=tau_z)
    elif reg_type == "center_weighted":
        return CenterWeightedRegularizer(
            device=device, tau=tau, tau_z=tau_z,
            center_weight=0.7, edge_weight=1.0
        )
    elif reg_type == "l1":
        return L1Regularizer(device=device, tau=tau_n)
    elif reg_type == "anisotropic":
        return AnisotropicDiffusionRegularizer(
            device=device, tau=tau, tau_z=tau_z, edge_threshold=0.1
        )
    else:  # hybrid or default
        return TV3DRegularizer(device=device, tau=tau, tau_z=tau_z)

def run_single_experiment(psf_stack_path, labels, params, experiment_name, logger):
    """단일 파라미터 조합으로 실험 실행"""
    mu1, mu2, mu3, tau, tau_z, tau_n, regularizer_type = params
    
    logger.info(f"실험 시작: {experiment_name}")
    logger.info(f"파라미터: μ1={mu1}, μ2={mu2}, μ3={mu3}, τ={tau:.0e}, τz={tau_z:.0e}, τn={tau_n}, reg={regularizer_type}")
    
    # ADMM 설정 생성
    config = config_3d.FIXED_CONFIG.copy()
    config.update({
        'psf_file': psf_stack_path,
        'img_file': config_3d.RAW_IMAGE_PATH,
        'save_dir': os.path.join(config_3d.RESULT_BASE_PATH, experiment_name),
        'mu1': mu1,
        'mu2': mu2,
        'mu3': mu3,
        'tau': tau,
        'tau_z': tau_z,
        'tau_n': tau_n,
    })
    
    # 저장 디렉토리 생성
    os.makedirs(config['save_dir'], exist_ok=True)
    
    try:
        # 정규화기 생성
        import torch
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        regularizer = create_regularizer(regularizer_type, device, tau, tau_z, tau_n)
        
        # ADMM 솔버 초기화
        solver = ADMM3D(config, regularizer=regularizer)
        
        # 실행
        start_time = time.time()
        final_reconstruction = solver.admm()
        elapsed_time = time.time() - start_time
        
        logger.info(f"실험 완료: {experiment_name} (소요시간: {elapsed_time:.1f}초)")
        
        # 결과 저장
        save_experiment_results(solver, config, params, experiment_name, logger)
        
        return True, elapsed_time
        
    except Exception as e:
        logger.error(f"실험 실패: {experiment_name} - {e}")
        return False, 0

def save_experiment_results(solver, config, params, experiment_name, logger):
    """실험 결과 저장"""
    result_dir = config['save_dir']
    
    # 파라미터 정보 저장
    param_info = {
        'mu1': params[0], 'mu2': params[1], 'mu3': params[2],
        'tau': params[3], 'tau_z': params[4], 'tau_n': params[5],
        'regularizer': params[6],
        'experiment_name': experiment_name,
        'config': config
    }
    
    import json
    with open(os.path.join(result_dir, 'parameters.json'), 'w') as f:
        json.dump(param_info, f, indent=2, default=str)
    
    # 수렴 메트릭 저장 (있다면)
    if hasattr(solver, 'plot_iteration_metrics'):
        try:
            solver.plot_iteration_metrics()
        except Exception as e:
            logger.warning(f"수렴 메트릭 저장 실패: {e}")
    
    logger.debug(f"결과 저장 완료: {result_dir}")

def run_parameter_sweep():
    """메인 파라미터 스위핑 함수"""
    logger = setup_logging()
    logger.info("=" * 70)
    logger.info("3D ADMM Parameter Sweep 시작")
    logger.info("=" * 70)
    
    # 디렉토리 생성
    os.makedirs(config_3d.RESULT_BASE_PATH, exist_ok=True)
    
    # PSF stack 준비 (한 번만)
    psf_stack_path, labels = prepare_fixed_psf(logger)
    
    # 파라미터 조합 수
    total_experiments = len(config_3d.PARAM_COMBINATIONS)
    logger.info(f"총 {total_experiments}개 실험 예정")
    
    # 각 파라미터 조합으로 실험
    successful_experiments = 0
    total_time = 0
    
    for i, params in enumerate(config_3d.PARAM_COMBINATIONS):
        mu1, mu2, mu3, tau, tau_z, tau_n, regularizer = params
        
        # 실험 이름 생성
        experiment_name = config_3d.get_experiment_name(mu1, mu2, mu3, tau, tau_z, tau_n, regularizer)
        
        logger.info(f"\n진행률: {i+1}/{total_experiments}")
        
        # 실험 실행
        success, elapsed_time = run_single_experiment(
            psf_stack_path, labels, params, experiment_name, logger
        )
        
        if success:
            successful_experiments += 1
            total_time += elapsed_time
    
    # 최종 결과
    logger.info("=" * 70)
    logger.info(f"파라미터 스위핑 완료!")
    logger.info(f"성공한 실험: {successful_experiments}/{total_experiments}")
    logger.info(f"총 소요 시간: {total_time:.1f}초")
    logger.info(f"평균 실험 시간: {total_time/(successful_experiments+1e-8):.1f}초 (성공한 실험 기준)")
    logger.info("=" * 70)

if __name__ == "__main__":
    run_parameter_sweep() 