# Diffusion Hyperprior 압축 모델 연구

이 프로젝트는 VAE 인코더를 통과한 latent space에서 하이퍼 인코더를 통해 z를 저장하고, 하이퍼 디코더 + diffusion denoising 과정을 통해 z에서 latent space를 예측하여 엔트로피 추정을 하는 새로운 이미지 압축 모델을 구현합니다.

## 🏗️ 아키텍처 개요

```
Input Image (x)
       ↓
   VAE Encoder (g_a)
       ↓
  Latent Space (y)
       ↓
  Hyper Encoder (h_a)
       ↓
   Hyperprior (z)
       ↓
Entropy Bottleneck
       ↓
   z_hat (compressed)
       ↓
  Hyper Decoder (h_s)
       ↓
   scales_hat
       ↓
Diffusion Denoising
       ↓
   y_hat (denoised)
       ↓
   VAE Decoder (g_s)
       ↓
Reconstructed Image (x_hat)
```

## 🔑 주요 특징

- **VAE 기반 압축**: CompressAI의 VAE 구조를 활용한 효율적인 latent space 압축
- **Hyperprior 모델**: Scale hyperprior를 통한 더 나은 엔트로피 모델링
- **Diffusion Denoising**: BK-SDM에서 영감을 받은 diffusion 과정을 통한 latent space 복원 품질 향상
- **엔트로피 추정**: Diffusion 과정을 통한 정확한 엔트로피 추정으로 압축 효율성 증대

## 📁 프로젝트 구조

```
.
├── diffusion_hyperprior_compression.py  # 메인 모델 구현
├── train_diffusion_hyperprior.py        # 학습 스크립트
├── evaluate_diffusion_hyperprior.py     # 평가 스크립트
├── requirements.txt                      # 의존성 패키지
├── README.md                            # 프로젝트 설명서
└── examples/                            # 사용 예제
    ├── train_example.py
    └── evaluate_example.py
```

## 🚀 설치 및 설정

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv diffusion_compression_env
source diffusion_compression_env/bin/activate  # Linux/Mac
# 또는
diffusion_compression_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. CompressAI 설치

```bash
# CompressAI 설치 (필요시)
git clone https://github.com/InterDigitalInc/CompressAI.git
cd CompressAI
pip install -e .
```

### 3. BK-SDM 모델 다운로드

```bash
# BK-SDM 모델 다운로드 (필요시)
# BK-SDM 폴더에 이미 포함되어 있음
```

## 🎯 사용법

### 1. 모델 학습

```bash
python train_diffusion_hyperprior.py \
    --data_path /path/to/training/data \
    --val_data_path /path/to/validation/data \
    --output_dir ./outputs \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --lambda_rate 1e-4 \
    --N 128 \
    --M 128 \
    --diffusion_steps 50 \
    --device cuda
```

#### 주요 파라미터 설명

- `--data_path`: 학습 데이터 경로
- `--val_data_path`: 검증 데이터 경로 (선택사항)
- `--output_dir`: 출력 디렉토리
- `--batch_size`: 배치 크기
- `--epochs`: 학습 에포크 수
- `--lr`: 학습률
- `--lambda_rate`: rate-distortion trade-off 파라미터
- `--N, --M`: 모델 채널 수
- `--diffusion_steps`: diffusion 단계 수
- `--device`: 사용할 디바이스 (cuda/cpu)

### 2. 모델 평가

```bash
python evaluate_diffusion_hyperprior.py \
    --model_path ./outputs/best_checkpoint.pth \
    --data_path /path/to/test/data \
    --output_dir ./evaluation_results \
    --batch_size 4 \
    --save_samples \
    --device cuda
```

#### 주요 파라미터 설명

- `--model_path`: 학습된 모델 경로
- `--data_path`: 테스트 데이터 경로
- `--output_dir`: 평가 결과 저장 디렉토리
- `--save_samples`: 샘플 이미지 저장 여부
- `--device`: 사용할 디바이스

### 3. Python 코드에서 직접 사용

```python
from diffusion_hyperprior_compression import create_diffusion_hyperprior_model

# 모델 생성
model = create_diffusion_hyperprior_model(
    N=128,           # VAE 채널 수
    M=128,           # Hyperprior 채널 수
    diffusion_steps=50  # Diffusion 단계 수
)

# 학습 모드
model.train()
# 또는 평가 모드
model.eval()

# Forward pass (학습)
output = model(input_images)
reconstructed = output['x_hat']
latent = output['y']
hyperprior = output['z']

# 압축
compressed = model.compress(input_images)
compressed_strings = compressed['strings']
compressed_shape = compressed['shape']

# 복원
decompressed = model.decompress(compressed_strings, compressed_shape)
reconstructed_image = decompressed['x_hat']
```

## 📊 성능 지표

모델은 다음 지표들을 통해 평가됩니다:

- **PSNR (Peak Signal-to-Noise Ratio)**: 이미지 품질 측정
- **MS-SSIM (Multi-Scale Structural Similarity)**: 구조적 유사성 측정
- **압축률**: 원본 대비 압축된 크기 비율
- **압축/복원 시간**: 처리 속도 측정
- **메모리 사용량**: 효율성 측정

## 🔬 연구 배경

### 1. 기존 연구와의 차별점

- **CompressAI**: 하이퍼프라이어 기반 압축의 기반 제공
- **BK-SDM**: Diffusion 모델의 효율적인 구조 활용
- **새로운 접근**: Diffusion denoising을 통한 latent space 복원 품질 향상

### 2. 핵심 아이디어

1. **VAE 인코더**: 입력 이미지를 효율적인 latent representation으로 변환
2. **하이퍼 인코더**: latent space의 통계적 특성을 모델링하는 z 생성
3. **Diffusion denoising**: z에서 y를 복원하는 과정에서 더 나은 품질 달성
4. **엔트로피 추정**: Diffusion 과정을 통한 정확한 압축 비트레이트 계산

## 🛠️ 커스터마이징

### 1. 모델 구조 변경

```python
# 채널 수 조정
model = create_diffusion_hyperprior_model(
    N=256,  # 더 큰 모델
    M=256,
    diffusion_steps=100  # 더 많은 diffusion 단계
)
```

### 2. 손실 함수 수정

```python
# train_diffusion_hyperprior.py에서 손실 함수 수정
# Rate-distortion trade-off 조정
lambda_rate = 1e-3  # 더 강한 압축
# 또는
lambda_rate = 1e-5  # 더 높은 품질
```

### 3. Diffusion 스케줄러 변경

```python
# diffusion_hyperprior_compression.py에서 스케줄러 수정
# Linear에서 Cosine으로 변경
self.noise_scheduler = DiffusionNoiseScheduler(
    num_train_timesteps=diffusion_steps,
    beta_start=0.0001,
    beta_end=0.02,
    schedule="cosine"  # "linear" 대신 "cosine" 사용
)
```

## 📈 실험 결과

### 1. 학습 과정 모니터링

```bash
# TensorBoard 사용
tensorboard --logdir ./outputs

# 또는 로그 파일 직접 확인
tail -f ./outputs/training.log
```

### 2. 결과 분석

- `./outputs/samples/`: 학습 중 생성된 샘플 이미지
- `./evaluation_results/`: 평가 결과 및 메트릭
- `./outputs/checkpoint_*.pth`: 체크포인트 파일들

## 🐛 문제 해결

### 1. 메모리 부족

```bash
# 배치 크기 줄이기
--batch_size 4

# 이미지 크기 줄이기
--image_size 128
```

### 2. 학습 불안정

```bash
# 학습률 줄이기
--lr 5e-5

# lambda_rate 조정
--lambda_rate 5e-5
```

### 3. CUDA 오류

```bash
# CPU 사용
--device cpu

# 또는 메모리 정리
torch.cuda.empty_cache()
```

## 🔮 향후 연구 방향

1. **다양한 데이터셋**: ImageNet, DIV2K 등 다양한 데이터셋에서의 성능 평가
2. **아키텍처 개선**: Transformer 기반 attention 메커니즘 도입
3. **효율성 향상**: 더 적은 diffusion 단계로 동일한 품질 달성
4. **실시간 압축**: 실시간 응용을 위한 속도 최적화
5. **비디오 압축**: 시계열 정보를 활용한 비디오 압축 확장

## 📚 참고 문헌

1. Ballé, J., et al. "Variational image compression with a scale hyperprior." ICLR 2018.
2. Rombach, R., et al. "High-resolution image synthesis with latent diffusion models." CVPR 2022.
3. BK-SDM: "Bottleneck Knowledge Distillation for Stable Diffusion Models"

## 🤝 기여하기

이 프로젝트에 기여하고 싶으시다면:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.

---

**참고**: 이 프로젝트는 연구 목적으로 개발되었으며, 실제 서비스에 적용하기 전에 충분한 테스트가 필요합니다. 