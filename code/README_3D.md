# 3D ADMM Parameter Sweep System
## 3D Reconstruction Parameter Sweeping Tool

이 시스템은 3D ADMM reconstruction을 위한 자동화된 파라미터 스위핑 도구입니다. PSF stack은 고정하고 다양한 ADMM 파라미터 조합을 테스트하여 최적의 복원 결과를 찾을 수 있습니다.

## 🏗️ 시스템 구조

```
3drecon/code/
├── config_3d.py              # 3D 파라미터 설정
├── parameter_sweep_3d.py     # 메인 실행 스크립트
├── plot_results_3d.py        # 결과 시각화
├── test_setup_3d.py          # 시스템 테스트
├── README_3D.md              # 이 문서
├── psf_preprocessing.py      # PSF 전처리 (기존)
├── admm_3d_refactored.py     # 3D ADMM 구현 (기존)
├── regularizers.py           # 정규화기들 (기존)
└── interactive_psf_selection.py  # 대화형 PSF 선택 (기존)
```

## 🚀 빠른 시작

### 1. 환경 설정

필요한 패키지들:
```bash
pip install numpy matplotlib scipy torch
```

### 2. 설정 파일 수정

`config_3d.py`에서 경로들을 실제 환경에 맞게 수정:

```python
# 기본 경로 설정
PSF_DIRECTORY = "your/psf/directory/path"
RAW_IMAGE_PATH = "your/raw/image/path.jpg"
RESULT_BASE_PATH = "your/results/directory/"
```

### 3. 시스템 테스트

```bash
python test_setup_3d.py
```

### 4. 파라미터 스위핑 실행

```bash
python parameter_sweep_3d.py
```

### 5. 결과 시각화

```bash
# 기본 요약 시각화
python plot_results_3d.py

# 상세 비교 시각화
python plot_results_3d.py --detailed --save detailed_results.png

# 파라미터 요약만 출력
python plot_results_3d.py --summary
```

## ⚙️ 설정 상세

### PSF 설정

```python
# PSF 선택 패턴 (당신의 기존 패턴 적용)
PSF_PRESET_CONFIG = {
    "start_index": 25,     # 시작 인덱스
    "step_size": 15,       # 그룹 크기
    "num_groups": 3        # 그룹 수 (총 45개 PSF)
}
```

### ADMM 파라미터 조합

```python
PARAM_COMBINATIONS = [
    # (mu1, mu2, mu3, tau, tau_z, tau_n, regularizer)
    (0.25, 0.68, 3.5, 6e-4, 6e-6, 0.06, 'center_weighted'),
    (0.5, 0.68, 4.0, 6e-4, 6e-5, 0.06, 'center_weighted'),
    (0.25, 0.5, 3.5, 8e-4, 6e-6, 0.08, '3dtv'),
    (0.5, 0.8, 4.0, 6e-4, 6e-5, 0.06, '3dtv'),
    (0.3, 0.68, 3.0, 1e-3, 1e-5, 0.1, 'l1'),
]
```

### 정규화기 옵션

- **'3dtv'**: 3D Total Variation - 표준 edge-preserving
- **'center_weighted'**: Center-Weighted TV - 치아 최적화 공간 정규화
- **'l1'**: L1 Sparsity - 희소성 촉진
- **'anisotropic'**: Anisotropic Diffusion - 고급 edge-preserving
- **'hybrid'**: 하이브리드 접근법

## 📊 결과 구조

각 실험은 다음과 같은 구조로 저장됩니다:

```
data/recon/3d_param_sweep/
├── mu1_0.25_mu2_0.68_mu3_3.5_tau_6e-04_tauz_6e-06_taun_0.06_center_weighted/
│   ├── parameters.json      # 파라미터 정보
│   ├── reconstruction_*.mat # 3D 복원 결과
│   ├── convergence/         # 수렴 메트릭
│   └── slices/             # 2D 슬라이스 이미지들
├── mu1_0.50_mu2_0.68_mu3_4.0_tau_6e-04_tauz_6e-05_taun_0.06_center_weighted/
│   └── ...
└── ...
```

## 🔧 고급 사용법

### 파라미터 조합 커스터마이징

범위 기반 자동 생성:
```python
MU1_RANGE = [0.25, 0.5, 0.75]
MU2_RANGE = [0.5, 0.68, 0.8] 
MU3_RANGE = [3.0, 3.5, 4.0]
TAU_RANGE = [6e-4, 8e-4, 1e-3]
REGULARIZER_RANGE = ['3dtv', 'center_weighted', 'l1']

# 자동 조합 생성
from itertools import product
PARAM_COMBINATIONS = list(product(MU1_RANGE, MU2_RANGE, MU3_RANGE, 
                                  TAU_RANGE, TAU_Z_RANGE, TAU_N_RANGE, 
                                  REGULARIZER_RANGE))
```

### 시각화 옵션

```bash
# 정규화기별 그룹화된 상세 비교
python plot_results_3d.py --detailed

# 특정 디렉토리의 결과 시각화
python plot_results_3d.py /path/to/your/results

# 이미지 저장
python plot_results_3d.py --save my_results.png

# 파라미터 요약 테이블만
python plot_results_3d.py --summary
```

## 📈 성능 및 최적화

### 메모리 관리
- PSF stack은 한 번만 로드하고 캐싱
- 각 실험 후 GPU 메모리 정리
- 큰 3D 볼륨은 압축하여 저장

### 실행 시간 추정
- PSF 전처리: ~30초 (캐시 사용시 즉시)
- 실험당 평균: 5-15분 (이터레이션 수에 따라)
- 5개 파라미터 조합: ~1시간

### 병렬 처리
현재는 순차 실행이지만, 다음과 같이 확장 가능:
```python
# 미래 개선사항: 멀티프로세싱
from multiprocessing import Pool
```

## 🐛 트러블슈팅

### 자주 발생하는 문제들

1. **GPU 메모리 부족**
   ```python
   # config_3d.py에서 수정
   FIXED_CONFIG['useGPU'] = False  # CPU 사용
   ```

2. **PSF 파일을 찾을 수 없음**
   ```python
   PSF_DIRECTORY = "/correct/path/to/psf/directory"
   ```

3. **Import 오류**
   ```bash
   # 필요한 파일들이 같은 디렉토리에 있는지 확인
   ls psf_preprocessing.py admm_3d_refactored.py regularizers.py
   ```

4. **복원 결과가 이상함**
   - 파라미터 범위 확인 (너무 큰/작은 값)
   - 정규화기 타입 확인
   - PSF와 raw 이미지 호환성 확인

### 로그 확인

```bash
tail -f parameter_sweep_3d.log
```

## 📚 관련 파일들

이 시스템은 다음 기존 모듈들을 활용합니다:

- **interactive_psf_selection.py**: 대화형 PSF 선택 도구
- **psf_preprocessing.py**: PSF 전처리 클래스
- **admm_3d_refactored.py**: 3D ADMM 구현
- **regularizers.py**: 다양한 정규화기 구현

## 🔄 워크플로우 비교

### 기존 워크플로우 (수동)
1. PSF 선택 → 2. 파라미터 설정 → 3. 실행 → 4. 결과 확인 → 5. 다른 파라미터로 반복

### 새로운 워크플로우 (자동)
1. 설정 파일 수정 → 2. `python parameter_sweep_3d.py` → 3. `python plot_results_3d.py` → 4. 모든 결과 한 번에 비교

## 🎯 향후 개선사항

- [ ] 멀티프로세싱 지원
- [ ] 웹 기반 결과 대시보드
- [ ] 자동 최적 파라미터 추천
- [ ] 실시간 진행상황 모니터링
- [ ] 클러스터/클라우드 실행 지원

---

문제가 있거나 개선사항이 있다면 언제든 말씀해주세요! 🚀 