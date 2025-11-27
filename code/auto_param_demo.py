#!/usr/bin/env python3
"""
자동 파라미터 최적화 데모 스크립트
=================================

3D ADMM reconstruction의 파라미터를 자동으로 최적화하는 방법들을 보여줍니다.

사용법:
    python auto_param_demo.py
"""

import os
import sys
import logging
from datetime import datetime
import config_3d


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'auto_param_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def demo_smart_grid_search():
    """스마트 그리드 탐색 데모"""
    print("=" * 60)
    print("🎯 스마트 그리드 탐색 데모")
    print("=" * 60)

    # 모드 설정
    config_3d.AUTO_PARAM_MODE = 'smart_grid'

    # 스마트 파라미터 조합 생성
    smart_combinations = config_3d.generate_smart_param_combinations()

    print(f"✅ 생성된 조합 수: {len(smart_combinations)}")
    print(f"📊 기존 수동 조합 수: {len(config_3d.PARAM_COMBINATIONS)}")

    # 처음 몇 개 조합 출력
    print("\n🔍 생성된 조합 예시:")
    for i, combo in enumerate(smart_combinations[:3]):
        mu1, mu2, mu3, tau, tau_z, tau_n, regularizer = combo
        print(f"  {i+1}. μ1={mu1:.2f}, μ2={mu2:.2f}, μ3={mu3:.1f}, "
              f"τ={tau:.0e}, τz={tau_z:.0e}, τn={tau_n:.2f}, reg={regularizer}")

    if len(smart_combinations) > 3:
        print(f"    ... (총 {len(smart_combinations)}개 조합)")

    return smart_combinations


def demo_adaptive_params():
    """적응적 파라미터 조정 데모"""
    print("\n" + "=" * 60)
    print("🧠 적응적 파라미터 조정 데모")
    print("=" * 60)

    # 기본 파라미터
    base_params = config_3d.get_data_adaptive_params()
    print(f"📋 기본 파라미터:")
    for param, value in base_params.items():
        if isinstance(value, float) and value < 1e-3:
            print(f"  {param}: {value:.0e}")
        else:
            print(f"  {param}: {value}")

    # 이미지 분석 기반 적응적 조정
    if os.path.exists(config_3d.RAW_IMAGE_PATH):
        print(f"\n🔍 이미지 분석 기반 조정:")
        print(f"  이미지 경로: {config_3d.RAW_IMAGE_PATH}")

        adapted_params = config_3d.get_data_adaptive_params(
            config_3d.RAW_IMAGE_PATH)

        print(f"\n📈 조정된 파라미터:")
        for param, value in adapted_params.items():
            if isinstance(value, float) and value < 1e-3:
                print(f"  {param}: {value:.0e}")
            else:
                print(f"  {param}: {value}")

        # 변경사항 표시
        print(f"\n🔄 변경사항:")
        for param in base_params:
            if param in adapted_params:
                base_val = base_params[param]
                adapted_val = adapted_params[param]
                if isinstance(base_val, (int, float)) and base_val != adapted_val:
                    change_ratio = adapted_val / base_val if base_val != 0 else 1
                    print(
                        f"  {param}: {base_val} → {adapted_val} ({change_ratio:.2f}x)")
    else:
        print(f"⚠️ 이미지 파일을 찾을 수 없습니다: {config_3d.RAW_IMAGE_PATH}")

    return base_params


def demo_autotune_settings():
    """자동 튜닝 설정 데모"""
    print("\n" + "=" * 60)
    print("⚙️ 자동 튜닝 설정 데모")
    print("=" * 60)

    # 현재 autotune 설정 표시
    fixed_config = config_3d.FIXED_CONFIG

    print(f"🔧 현재 자동 튜닝 설정:")
    autotune_keys = [
        'autotune', 'autotune_start_iter', 'autotune_interval',
        'mu_inc', 'mu_dec', 'resid_tol', 'adaptive_tau',
        'tau_adaptation_rate', 'convergence_window', 'min_improvement_threshold'
    ]

    for key in autotune_keys:
        if key in fixed_config:
            print(f"  {key}: {fixed_config[key]}")

    # 자동 튜닝 작동 방식 설명
    print(f"\n📖 자동 튜닝 작동 방식:")
    print(f"  1. {fixed_config['autotune_start_iter']}번 반복 후 자동 튜닝 시작")
    print(f"  2. {fixed_config['autotune_interval']}번마다 파라미터 조정 검토")
    print(
        f"  3. residual 비율에 따라 μ 값들을 {fixed_config['mu_inc']:.2f}배 증가 또는 {fixed_config['mu_dec']:.2f}배 감소")
    print(
        f"  4. tolerance: {fixed_config['resid_tol']:.1f} (primal/dual residual 비율)")

    if fixed_config.get('adaptive_tau', False):
        print(
            f"  5. τ 값도 적응적으로 조정 (조정 비율: {fixed_config['tau_adaptation_rate']:.2f})")

    return fixed_config


def demo_comparison():
    """최적화 방법 비교 데모"""
    print("\n" + "=" * 60)
    print("📊 최적화 방법 비교")
    print("=" * 60)

    methods = {
        'manual': "수동 파라미터 조합",
        'smart_grid': "스마트 그리드 탐색",
        'adaptive': "적응적 파라미터 조정",
        'autotune': "실시간 자동 튜닝"
    }

    advantages = {
        'manual': [
            "완전한 제어 가능",
            "검증된 파라미터 사용",
            "예측 가능한 결과"
        ],
        'smart_grid': [
            "효율적인 파라미터 공간 탐색",
            "다양한 조합 자동 생성",
            "중복 제거 및 최적화"
        ],
        'adaptive': [
            "데이터 특성 자동 분석",
            "노이즈/대비 레벨 고려",
            "초기 파라미터 자동 조정"
        ],
        'autotune': [
            "실시간 수렴 모니터링",
            "μ 값 동적 조정",
            "수렴 속도 향상"
        ]
    }

    for method, description in methods.items():
        print(f"\n🔹 {method.upper()}: {description}")
        for advantage in advantages[method]:
            print(f"  ✅ {advantage}")

    print(f"\n💡 권장 사용 방법:")
    print(f"  1. 새로운 데이터셋 → 'adaptive' + 'autotune' 조합")
    print(f"  2. 파라미터 탐색 → 'smart_grid' 먼저 실행")
    print(f"  3. 정밀한 조정 → 'manual' 파라미터 세팅")
    print(f"  4. 실시간 최적화 → 'autotune' 활성화")


def run_optimization_demo():
    """통합 최적화 데모 실행"""
    print("\n" + "=" * 60)
    print("🚀 통합 최적화 데모 실행")
    print("=" * 60)

    logger = setup_logging()
    logger.info("자동 파라미터 최적화 데모 시작")

    # 1. 적응적 파라미터로 초기 설정
    print("\n1️⃣ 적응적 파라미터 분석...")
    adaptive_params = config_3d.get_data_adaptive_params(
        config_3d.RAW_IMAGE_PATH)

    # 2. 스마트 그리드로 추가 탐색
    print("\n2️⃣ 스마트 그리드 탐색...")
    config_3d.AUTO_PARAM_MODE = 'smart_grid'
    smart_combinations = config_3d.generate_smart_param_combinations()

    # 3. 최적 조합 선택 (예시: 첫 번째 조합)
    if smart_combinations:
        best_combo = smart_combinations[0]
        mu1, mu2, mu3, tau, tau_z, tau_n, regularizer = best_combo

        print(f"\n3️⃣ 선택된 최적 파라미터:")
        print(f"  μ1={mu1:.2f}, μ2={mu2:.2f}, μ3={mu3:.1f}")
        print(f"  τ={tau:.0e}, τz={tau_z:.0e}, τn={tau_n:.2f}")
        print(f"  regularizer={regularizer}")

        # 4. 실제 실행을 위한 설정 생성
        optimized_config = config_3d.FIXED_CONFIG.copy()
        optimized_config.update({
            'mu1': mu1,
            'mu2': mu2,
            'mu3': mu3,
            'tau': tau,
            'tau_z': tau_z,
            'tau_n': tau_n,
            'regularizer': regularizer,
            'autotune': 1,  # 실시간 튜닝도 활성화
        })

        print(f"\n4️⃣ 최적화된 설정 준비 완료!")
        print(f"  자동 튜닝: {'ON' if optimized_config['autotune'] else 'OFF'}")
        print(f"  최대 반복: {optimized_config['max_iter']}")

        logger.info(
            f"최적화 완료: {config_3d.get_experiment_name(mu1, mu2, mu3, tau, tau_z, tau_n, regularizer)}")

        return optimized_config

    else:
        print("⚠️ 스마트 그리드 탐색 결과가 없습니다.")
        return None


def main():
    """메인 데모 실행"""
    print("🎉 3D ADMM 자동 파라미터 최적화 데모")
    print("=" * 60)

    # 각 기능 데모 실행
    demo_smart_grid_search()
    demo_adaptive_params()
    demo_autotune_settings()
    demo_comparison()

    # 통합 데모
    optimized_config = run_optimization_demo()

    if optimized_config:
        print(f"\n🎯 최종 결과:")
        print(f"  ✅ 자동 파라미터 최적화 완료")
        print(f"  📂 설정 파일: config_3d.py")
        print(f"  🚀 실행 방법: python parameter_sweep_3d.py")

        # 사용 예시 코드 출력
        print(f"\n💻 사용 예시:")
        print(f"```python")
        print(f"# 1. 스마트 그리드 모드 활성화")
        print(f"config_3d.AUTO_PARAM_MODE = 'smart_grid'")
        print(f"")
        print(f"# 2. 자동 생성된 파라미터 조합 사용")
        print(f"combinations = config_3d.generate_smart_param_combinations()")
        print(f"")
        print(f"# 3. 적응적 파라미터로 시작")
        print(f"adaptive_params = config_3d.get_data_adaptive_params()")
        print(f"")
        print(f"# 4. 자동 튜닝 활성화")
        print(f"config_3d.FIXED_CONFIG['autotune'] = 1")
        print(f"```")

    print(f"\n🎊 데모 완료!")


if __name__ == "__main__":
    main()
