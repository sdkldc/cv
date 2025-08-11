#!/usr/bin/env python3
"""
BK-SDM 기반 압축 모델 평가 예제

이 스크립트는 학습된 모델을 평가하는 방법을 보여줍니다.
"""

import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import tempfile
import shutil
import json

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bk_sdm_compression import create_bk_sdm_compression_model


def create_test_dataset(num_images=20, image_size=64):
    """테스트용 데이터셋 생성"""
    temp_dir = tempfile.mkdtemp()
    images_dir = os.path.join(temp_dir, 'test', 'class1')
    os.makedirs(images_dir, exist_ok=True)
    
    # 테스트 이미지 생성 (더 복잡한 패턴)
    for i in range(num_images):
        # 체커보드 패턴과 그라데이션 조합
        img = torch.zeros(3, image_size, image_size)
        
        # 체커보드 패턴
        checkerboard = torch.zeros(image_size, image_size)
        for x in range(image_size):
            for y in range(image_size):
                if (x // 8 + y // 8) % 2 == 0:
                    checkerboard[x, y] = 1
        
        # 각 채널에 다른 패턴 적용
        img[0] = checkerboard * 0.8 + torch.rand(image_size, image_size) * 0.2
        img[1] = (1 - checkerboard) * 0.6 + torch.rand(image_size, image_size) * 0.4
        img[2] = torch.rand(image_size, image_size)
        
        img = (img * 255).byte()
        img_pil = torchvision.transforms.ToPILImage()(img)
        
        img_path = os.path.join(images_dir, f'test_{i:04d}.png')
        img_pil.save(img_path)
    
    return temp_dir, images_dir


def evaluate_example():
    """평가 예제 실행"""
    print("=== BK-SDM 기반 압축 모델 평가 예제 ===")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 경로 확인
    model_path = "./example_outputs/example_model.pth"
    if not os.path.exists(model_path):
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 train_example.py를 실행하여 모델을 학습하세요.")
        return
    
    # BK-SDM 경로 설정 (예제용 더미 경로)
    bk_sdm_path = "../BK-SDM"  # 실제 경로로 수정 필요
    
    # 테스트 데이터셋 생성
    print("테스트 데이터셋 생성 중...")
    temp_dir, images_dir = create_test_dataset(num_images=20, image_size=64)
    print(f"테스트 데이터셋 생성 완료: {images_dir}")
    
    try:
        # 모델 생성 및 로드 (커스텀 컴포넌트 사용)
        print("모델 로드 중...")
        model = create_bk_sdm_compression_model(
            bk_sdm_path=bk_sdm_path,
            N=64,
            M=64,
            diffusion_steps=15,
            use_pretrained_vae=False,
            use_pretrained_unet=False
        ).to(device)
        
        # 학습된 가중치 로드
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("모델 로드 완료")
        
        # 평가 모드로 설정
        model.eval()
        
        # 데이터 로더 생성
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        dataset = torchvision.datasets.ImageFolder(temp_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        print(f"테스트 데이터 로더 생성 완료: {len(dataset)} 이미지")
        
        # 평가 실행
        print("모델 평가 시작...")
        
        total_psnr = 0
        total_compression_ratio = 0
        total_compression_time = 0
        total_decompression_time = 0
        total_original_size = 0
        total_compressed_size = 0
        
        import time
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                batch_size = data.shape[0]
                
                print(f"배치 {batch_idx + 1}/{len(dataloader)} 처리 중...")
                
                # 원본 크기 계산
                original_size = data.numel() * data.element_size()
                total_original_size += original_size
                
                # 압축
                start_time = time.time()
                compressed = model.compress(data)
                compression_time = time.time() - start_time
                total_compression_time += compression_time
                
                # 압축된 크기 계산
                compressed_size = sum(len(s) for s in compressed['strings'])
                total_compressed_size += compressed_size
                
                # 압축률 계산
                compression_ratio = original_size / compressed_size
                total_compression_ratio += compression_ratio
                
                # 복원
                start_time = time.time()
                decompressed = model.decompress(compressed['strings'], compressed['shape'])
                decompression_time = time.time() - start_time
                total_decompression_time += decompression_time
                
                # 품질 측정
                mse = torch.nn.functional.mse_loss(data, decompressed['x_hat'])
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr
                
                print(f"  PSNR: {psnr:.2f} dB, 압축률: {compression_ratio:.2f}x")
                print(f"  압축 시간: {compression_time:.4f}s, 복원 시간: {decompression_time:.4f}s")
                
                # 샘플 이미지 저장 (첫 번째 배치만)
                if batch_idx == 0:
                    output_dir = "./example_outputs/samples"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 원본과 복원 이미지 비교
                    comparison = torch.cat([
                        data[:4],  # 원본
                        decompressed['x_hat'][:4]  # 복원
                    ], dim=0)
                    
                    torchvision.utils.save_image(
                        comparison,
                        os.path.join(output_dir, "comparison.png"),
                        nrow=4,
                        normalize=True
                    )
                    print(f"  샘플 이미지 저장: {output_dir}/comparison.png")
        
        # 전체 결과 계산
        num_batches = len(dataloader)
        avg_psnr = total_psnr / num_batches
        avg_compression_ratio = total_compression_ratio / num_batches
        avg_compression_time = total_compression_time / num_batches
        avg_decompression_time = total_decompression_time / num_batches
        overall_compression_ratio = total_original_size / total_compressed_size
        
        # 결과 출력
        print("\n=== 평가 결과 요약 ===")
        print(f"평균 PSNR: {avg_psnr:.2f} dB")
        print(f"평균 압축률: {avg_compression_ratio:.2f}x")
        print(f"전체 압축률: {overall_compression_ratio:.2f}x")
        print(f"평균 압축 시간: {avg_compression_time:.4f}초")
        print(f"평균 복원 시간: {avg_decompression_time:.4f}초")
        print(f"원본 총 크기: {total_original_size / 1024:.2f} KB")
        print(f"압축 총 크기: {total_compressed_size / 1024:.2f} KB")
        
        # 결과를 JSON 파일로 저장
        results = {
            'metrics': {
                'avg_psnr': avg_psnr,
                'avg_compression_ratio': avg_compression_ratio,
                'overall_compression_ratio': overall_compression_ratio,
                'avg_compression_time': avg_compression_time,
                'avg_decompression_time': avg_decompression_time,
            },
            'totals': {
                'total_original_size': total_original_size,
                'total_compressed_size': total_compressed_size,
                'total_compression_time': total_compression_time,
                'total_decompression_time': total_decompression_time,
            },
            'model_info': {
                'N': 64,
                'M': 64,
                'diffusion_steps': 15,
                'parameters': sum(p.numel() for p in model.parameters()),
            }
        }
        
        results_path = "./example_outputs/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n상세 결과 저장: {results_path}")
        print("=== 평가 완료 ===")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        print("BK-SDM 경로를 확인하거나 커스텀 컴포넌트를 사용하세요.")
        
    finally:
        # 임시 파일 정리
        print("임시 파일 정리 중...")
        shutil.rmtree(temp_dir)
        print("정리 완료")


if __name__ == '__main__':
    evaluate_example() 