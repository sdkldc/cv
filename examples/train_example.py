#!/usr/bin/env python3
"""
Diffusion Hyperprior 압축 모델 학습 예제

이 스크립트는 간단한 예제 데이터셋을 사용하여 모델을 학습하는 방법을 보여줍니다.
"""

import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import tempfile
import shutil

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_hyperprior_compression import create_diffusion_hyperprior_model


def create_sample_dataset(num_images=100, image_size=64):
    """간단한 샘플 데이터셋 생성"""
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    images_dir = os.path.join(temp_dir, 'images', 'class1')
    os.makedirs(images_dir, exist_ok=True)
    
    # 간단한 이미지 생성 (색상 그라데이션)
    for i in range(num_images):
        # 랜덤 색상 그라데이션 이미지 생성
        img = torch.rand(3, image_size, image_size)
        img = (img * 255).byte()
        
        # PIL 이미지로 변환
        img_pil = torchvision.transforms.ToPILImage()(img)
        
        # 저장
        img_path = os.path.join(images_dir, f'image_{i:04d}.png')
        img_pil.save(img_path)
    
    return temp_dir, images_dir


def train_example():
    """학습 예제 실행"""
    print("=== Diffusion Hyperprior 압축 모델 학습 예제 ===")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 샘플 데이터셋 생성
    print("샘플 데이터셋 생성 중...")
    temp_dir, images_dir = create_sample_dataset(num_images=50, image_size=64)
    print(f"데이터셋 생성 완료: {images_dir}")
    
    try:
        # 모델 생성
        print("모델 생성 중...")
        model = create_diffusion_hyperprior_model(
            N=64,              # 작은 모델 (예제용)
            M=64,
            diffusion_steps=20  # 적은 diffusion 단계 (예제용)
        ).to(device)
        
        print(f"모델 생성 완료: {sum(p.numel() for p in model.parameters()):,} 파라미터")
        
        # 데이터 로더 생성
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        dataset = torchvision.datasets.ImageFolder(temp_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"데이터 로더 생성 완료: {len(dataset)} 이미지")
        
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # 학습 루프 (간단한 예제)
        print("학습 시작...")
        model.train()
        
        for epoch in range(3):  # 3 에포크만 실행 (예제용)
            total_loss = 0
            
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                
                # 손실 계산 (간단한 reconstruction loss)
                loss = torch.nn.functional.mse_loss(output['x_hat'], data)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.6f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} 완료, 평균 손실: {avg_loss:.6f}")
        
        print("학습 완료!")
        
        # 모델 저장
        output_dir = "./example_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(model.state_dict(), os.path.join(output_dir, "example_model.pth"))
        print(f"모델 저장 완료: {output_dir}/example_model.pth")
        
        # 간단한 테스트
        print("모델 테스트 중...")
        model.eval()
        
        with torch.no_grad():
            test_batch = next(iter(dataloader))[0][:2].to(device)  # 2개 이미지만 테스트
            
            # 압축
            compressed = model.compress(test_batch)
            print(f"압축 완료: {len(compressed['strings'])} 문자열")
            
            # 복원
            decompressed = model.decompress(compressed['strings'], compressed['shape'])
            print(f"복원 완료: {decompressed['x_hat'].shape}")
            
            # 품질 측정
            mse = torch.nn.functional.mse_loss(test_batch, decompressed['x_hat'])
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            print(f"테스트 결과 - MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
        
        print("=== 예제 실행 완료 ===")
        
    finally:
        # 임시 파일 정리
        print("임시 파일 정리 중...")
        shutil.rmtree(temp_dir)
        print("정리 완료")


if __name__ == "__main__":
    train_example() 