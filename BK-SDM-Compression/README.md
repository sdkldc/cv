# BK-SDM 기반 이미지 압축 모델

이 프로젝트는 BK-SDM의 pre-trained 인코더/디코더와 Denoise U-Net을 활용하여 이미지 압축을 수행하는 혁신적인 모델입니다. Hyper-encoder를 통과한 z를 diffusion denoising의 초기 latent space로 사용하고, BK-SDM의 pre-trained U-Net으로 latent space를 예측하여 압축 저장과 복원을 수행합니다.

## 🏗️ 아키텍처 개요

```
Input Image (x)
       ↓
BK-SDM VAE Encoder (pre-trained)
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
BK-SDM Denoise U-Net (pre-trained)
       ↓
   y_hat (denoised)
       ↓
BK-SDM VAE Decoder (pre-trained)
       ↓
Reconstructed Image (x_hat)
```

## 🔑 주요 특징

- **BK-SDM VAE 활용**: Pre-trained된 BK-SDM의 VAE 인코더/디코더를 사용하여 고품질 latent representation 생성
- **Hyperprior 모델**: Scale hyperprior를 통한 효율적인 엔트로피 모델링
- **BK-SDM U-Net**: Pre-trained된 BK-SDM의 Denoise U-Net을 활용한 diffusion denoising
- **하이브리드 접근**: Pre-trained 모델과 학습 가능한 컴포넌트의 조합으로 최적의 성능 달성

## 📁 프로젝트 구조

```
BK-SDM-Compression/
├── bk_sdm_compression.py           # 메인 모델 구현
├── train_bk_sdm_compression.py     # 학습 스크립트
├── evaluate_bk_sdm_compression.py  # 평가 스크립트
├── requirements.txt                 # 의존성 패키지
├── README.md                       # 프로젝트 설명서
└── examples/                       # 사용 예제
    ├── train_example.py
    └── evaluate_example.py
```

## 🚀 설치 및 설정

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv bk_sdm_compression_env
source bk_sdm_compression_env/bin/activate  # Linux/Mac
# 또는
bk_sdm_compression_env\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. BK-SDM 모델 준비

```bash
# BK-SDM 모델이 이미 준비되어 있어야 함
# BK-SDM 폴더 경로를 확인하고 설정
```

### 3. CompressAI 설치

```bash
# CompressAI 설치 (필요시)
git clone https://github.com/InterDigitalInc/CompressAI.git
cd CompressAI
pip install -e .
```

## 🎯 사용법

### 1. 모델 학습

```bash
python train_bk_sdm_compression.py \
    --bk_sdm_path ../BK-SDM \
    --data_path /path/to/training/data \
    --val_data_path /path/to/validation/data \
    --output_dir ./outputs \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --lambda_rate 1e-4 \
    --N 128 \
    --M 128 \
    --diffusion_steps 30 \
    --device cuda \
    --use_pretrained_vae \
    --use_pretrained_unet
```

#### 주요 파라미터 설명

- `--bk_sdm_path`: BK-SDM 모델 디렉토리 경로
- `--data_path`: 학습 데이터 경로
- `--val_data_path`: 검증 데이터 경로 (선택사항)
- `--output_dir`: 출력 디렉토리
- `--batch_size`: 배치 크기 (메모리 제약으로 4 권장)
- `--epochs`: 학습 에포크 수
- `--lr`: 학습률
- `--lambda_rate`: rate-distortion trade-off 파라미터
- `--N, --M`: 모델 채널 수
- `--diffusion_steps`: diffusion 단계 수
- `--use_pretrained_vae`: BK-SDM VAE 사용 여부
- `--use_pretrained_unet`: BK-SDM U-Net 사용 여부

### 2. 모델 평가

```bash
python evaluate_bk_sdm_compression.py \
    --model_path ./outputs/best_checkpoint.pth \
    --bk_sdm_path ../BK-SDM \
    --data_path /path/to/test/data \
    --output_dir ./evaluation_results \
    --batch_size 2 \
    --save_samples \
    --device cuda \
    --use_pretrained_vae \
    --use_pretrained_unet
```

### 3. Python 코드에서 직접 사용

```python
from bk_sdm_compression import create_bk_sdm_compression_model

# 모델 생성
model = create_bk_sdm_compression_model(
    bk_sdm_path="../BK-SDM",
    N=128,           # VAE 채널 수
    M=128,           # Hyperprior 채널 수
    diffusion_steps=30,  # Diffusion 단계 수
    use_pretrained_vae=True,    # BK-SDM VAE 사용
    use_pretrained_unet=True    # BK-SDM U-Net 사용
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

- **BK-SDM**: Pre-trained된 고품질 VAE와 U-Net 활용
- **CompressAI**: 하이퍼프라이어 기반 압축의 기반 제공
- **새로운 접근**: Pre-trained 모델과 학습 가능한 컴포넌트의 하이브리드 구조

### 2. 핵심 아이디어

1. **BK-SDM VAE**: Pre-trained된 고품질 인코더/디코더로 latent space 생성
2. **하이퍼 인코더**: latent space의 통계적 특성을 모델링하는 z 생성
3. **BK-SDM U-Net**: Pre-trained된 U-Net으로 diffusion denoising 수행
4. **엔트로피 추정**: 하이퍼프라이어를 통한 정확한 압축 비트레이트 계산

## 🛠️ 커스터마이징

### 1. Pre-trained 컴포넌트 사용 여부

```bash
# VAE와 U-Net 모두 pre-trained 사용
--use_pretrained_vae --use_pretrained_unet

# 커스텀 컴포넌트만 사용
# (플래그 없이 실행)
```

### 2. 모델 구조 변경

```python
# 채널 수 조정
model = create_bk_sdm_compression_model(
    bk_sdm_path="../BK-SDM",
    N=256,  # 더 큰 모델
    M=256,
    diffusion_steps=50  # 더 많은 diffusion 단계
)
```

### 3. Diffusion 스케줄러 변경

```python
# bk_sdm_compression.py에서 스케줄러 수정
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
--batch_size 2

# 이미지 크기 줄이기
--image_size 128

# Diffusion 단계 줄이기
--diffusion_steps 20
```

### 2. BK-SDM 로딩 실패

```bash
# Pre-trained 컴포넌트 사용하지 않기
# (플래그 없이 실행하면 커스텀 컴포넌트 사용)

# 또는 BK-SDM 경로 확인
--bk_sdm_path /correct/path/to/BK-SDM
```

### 3. 학습 불안정

```bash
# 학습률 줄이기
--lr 5e-5

# lambda_rate 조정
--lambda_rate 5e-5
```

## 🔮 향후 연구 방향

1. **다양한 Pre-trained 모델**: 다른 diffusion 모델들과의 호환성 확장
2. **효율성 향상**: 더 적은 diffusion 단계로 동일한 품질 달성
3. **실시간 압축**: 실시간 응용을 위한 속도 최적화
4. **비디오 압축**: 시계열 정보를 활용한 비디오 압축 확장
5. **적응형 압축**: 입력 이미지 특성에 따른 동적 압축률 조정

## 📚 참고 문헌

1. Ballé, J., et al. "Variational image compression with a scale hyperprior." ICLR 2018.
2. Rombach, R., et al. "High-resolution image synthesis with latent diffusion models." CVPR 2022.
3. BK-SDM: "Bottleneck Knowledge Distillation for Stable Diffusion Models"
4. CompressAI: "PyTorch-based image compression framework"

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

**참고**: 이 프로젝트는 BK-SDM 모델이 필요하며, 연구 목적으로 개발되었습니다. 실제 서비스에 적용하기 전에 충분한 테스트가 필요합니다. 