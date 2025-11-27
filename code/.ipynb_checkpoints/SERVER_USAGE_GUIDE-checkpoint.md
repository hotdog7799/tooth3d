# 서버에서 파라미터 최적화 실행 가이드

공용 서버에서 효율적으로 파라미터 최적화를 실행하는 방법을 설명합니다.

## 🖥️ 서버 실행 명령어 완벽 가이드

### 기본 명령어 설명

#### 1. `nohup` 이란?
- **no hang up**의 줄임말
- 터미널 연결이 끊어져도 프로그램이 계속 실행되도록 함
- SSH 연결이 끊어져도 프로그램이 백그라운드에서 계속 돌아감

#### 2. `>` 와 `2>&1` 설명
```bash
python run_optimization.py > optimization.log 2>&1 &
```

- `>` : 표준 출력(stdout)을 파일로 리다이렉션
- `2>&1` : 에러 출력(stderr)도 같은 파일로 리다이렉션
  - `2` = stderr (에러 메시지)
  - `&1` = stdout과 같은 곳으로 (위에서 지정한 파일)
- `&` : 백그라운드에서 실행

**결과**: 모든 출력과 에러가 `optimization.log` 파일에 저장되면서 백그라운드 실행

## 🚀 추천 실행 방법들

### 방법 1: nohup 사용 (가장 추천)
```bash
nohup python run_optimization.py > optimization.log 2>&1 &
```

**장점:**
- SSH 연결이 끊어져도 계속 실행
- 모든 로그가 파일에 저장됨
- 간단하고 안정적

**사용법:**
1. 명령어 실행
2. `[1] 12345` 같은 프로세스 번호가 나오면 성공
3. `exit`로 SSH 종료해도 프로그램은 계속 실행

### 방법 2: tmux 사용 (고급 사용자)
```bash
# tmux 세션 생성 및 시작
tmux new-session -d -s optimization 'python run_optimization.py'

# 진행 상황 확인
tmux attach-session -t optimization
```

**장점:**
- 실시간 모니터링 가능
- 여러 창 동시 관리
- 더 많은 제어 옵션

### 방법 3: screen 사용
```bash
# screen 세션 시작
screen -S optimization
python run_optimization.py

# Ctrl+A, D로 detach
# screen -r optimization으로 다시 attach
```

## 📊 진행 상황 모니터링

### 1. 로그 파일 실시간 확인
```bash
tail -f optimization.log
```

### 2. 프로세스 확인
```bash
ps aux | grep python
```

### 3. GPU 사용량 확인
```bash
nvidia-smi
```

### 4. 메모리 사용량 확인
```bash
free -h
htop
```

## 🗂️ 결과 파일 구조

최적화 완료 후 생성되는 파일들:

```
optimization_results_20250101_120000/
├── optimization.log                    # 실행 로그
├── optimization_results.csv            # 📊 분석용 CSV (Excel에서 열기 좋음)
├── parameter_analysis.csv              # 📈 파라미터 영향도 분석
├── top_10_results.csv                  # 🏆 최고 성능 10개 결과
├── best_parameters.json                # 🎯 최적 파라미터
├── OPTIMIZATION_SUMMARY.txt            # 📋 전체 요약
├── detailed_results.json               # 🔍 상세 결과 (개발자용)
└── exp_XXXX_XXXXXXXXXX/                # 각 실험 폴더
    ├── projections_iter_FINAL.png      # 📸 마지막 iteration 투영 이미지
    ├── slices_iter_FINAL.png           # 📸 마지막 iteration 슬라이스 이미지
    └── convergence_metrics.png         # 📈 수렴 그래프
```

**중요:** `.mat` 파일은 저장되지 않아서 용량 절약! (각 실험당 ~10MB 대신 100MB+)

## 📈 결과 분석 방법

### 1. Excel에서 분석
```bash
# CSV 파일을 로컬로 다운로드
scp user@server:/path/to/optimization_results.csv ./
```

Excel에서 `optimization_results.csv` 열어서:
- 파라미터별 성능 비교
- 그래프 생성
- 필터링/정렬

### 2. Python에서 분석
```python
import pandas as pd

# 결과 로드
df = pd.read_csv('optimization_results.csv')

# 최고 성능 결과들
top_results = df.nlargest(5, 'metric_composite_score')
print(top_results)

# 파라미터 영향도 확인
param_analysis = pd.read_csv('parameter_analysis.csv')
print(param_analysis)
```

### 3. 최적 파라미터 바로 확인
```bash
cat best_parameters.json
```

## 🛠️ 트러블슈팅

### 문제 1: 프로그램이 중단됨
**해결:** 자동으로 재시작 가능
```bash
# 같은 명령어로 다시 실행하면 중단된 지점부터 계속
nohup python run_optimization.py > optimization.log 2>&1 &
```

### 문제 2: GPU 메모리 부족
**현상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
1. 실행 중인 다른 GPU 프로세스 확인: `nvidia-smi`
2. 불필요한 프로세스 종료
3. 메모리 사용량 줄이기 (코드에서 자동으로 정리함)

### 문제 3: 디스크 공간 부족
**확인:**
```bash
df -h
```

**해결:**
- 불필요한 파일 삭제
- 실험 결과를 외부로 백업 후 삭제

### 문제 4: 프로세스 강제 종료 필요
```bash
# 프로세스 ID 찾기
ps aux | grep python

# 프로세스 종료 (PID는 위에서 확인한 번호)
kill PID

# 강제 종료가 필요한 경우
kill -9 PID
```

## ⚡ 성능 최적화 팁

### 1. 메모리 관리
- 코드에서 자동으로 메모리 정리함 (`gc.collect()`)
- GPU 메모리도 자동 정리 (`torch.cuda.empty_cache()`)

### 2. 동시 실행 피하기
- 한 번에 하나의 최적화만 실행 권장
- 여러 개 실행시 GPU 메모리 부족 가능

### 3. 실행 시간 예측
- Quick Test: ~30분 (8개 조합)
- Standard: ~2-4시간 (54개 조합)  
- Thorough: ~6-12시간 (256개 조합)

### 4. 최적 실행 시간
- 밤이나 주말에 실행 (서버 부하 적음)
- 긴 최적화는 금요일 저녁에 시작

## 📋 실행 체크리스트

최적화 실행 전 확인사항:

- [ ] PSF 파일과 raw image 파일 경로 확인
- [ ] 디스크 공간 충분한지 확인 (최소 20GB 권장)
- [ ] 다른 GPU 프로세스와 충돌하지 않는지 확인
- [ ] 실행 시간 예상하고 적절한 강도 선택
- [ ] nohup으로 실행해서 SSH 연결 끊어져도 안전하게

## 🎯 최종 실행 명령어

**추천 실행 방법:**
```bash
# 최적화 시작
nohup python run_optimization.py > optimization.log 2>&1 &

# 진행 상황 확인
tail -f optimization.log

# 완료 후 결과 확인
cat best_parameters.json
```

이제 서버에서 안전하고 효율적으로 파라미터 최적화를 실행할 수 있습니다! 🚀 