import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import logging
from tqdm import tqdm
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bk_sdm_compression import create_bk_sdm_compression_model


def setup_logging(log_dir: str):
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def create_dataloader(data_path: str, batch_size: int, image_size: int = 256):
    """데이터 로더 생성"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


def compute_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """PSNR 계산"""
    mse = torch.mean((x - x_hat) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_msssim(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """MS-SSIM 계산 (간단한 버전)"""
    # 실제 구현에서는 MS-SSIM 라이브러리 사용 권장
    return torch.mean(torch.abs(x - x_hat)).item()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    lambda_rate: float = 1e-4
):
    """한 에포크 학습"""
    model.train()
    total_loss = 0
    total_psnr = 0
    total_msssim = 0
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, _) in enumerate(progress_bar):
        data = data.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(output['x_hat'], data)
        
        # Rate loss (entropy)
        rate_loss = 0
        if output['likelihoods']['z'] is not None:
            rate_loss = -torch.mean(output['likelihoods']['z'])
        
        # Total loss
        loss = recon_loss + lambda_rate * rate_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            psnr = compute_psnr(data, output['x_hat'])
            msssim = compute_msssim(data, output['x_hat'])
        
        total_loss += loss.item()
        total_psnr += psnr
        total_msssim += msssim
        
        # Progress bar update
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'PSNR': f'{psnr:.2f}',
            'MS-SSIM': f'{msssim:.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    avg_msssim = total_msssim / len(dataloader)
    
    return avg_loss, avg_psnr, avg_msssim


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    epoch: int
):
    """검증 및 샘플 이미지 저장"""
    model.eval()
    total_psnr = 0
    total_msssim = 0
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= 5:  # 처음 5개 배치만 처리
                break
                
            data = data.to(device)
            
            # Forward pass
            output = model(data)
            
            # Metrics
            psnr = compute_psnr(data, output['x_hat'])
            msssim = compute_msssim(data, output['x_hat'])
            
            total_psnr += psnr
            total_msssim += msssim
            
            # Save sample images
            if batch_idx == 0:
                save_image(
                    torch.cat([data[:4], output['x_hat'][:4]], dim=0),
                    os.path.join(save_dir, f'epoch_{epoch}_samples.png'),
                    nrow=4,
                    normalize=True
                )
    
    avg_psnr = total_psnr / min(5, len(dataloader))
    avg_msssim = total_msssim / min(5, len(dataloader))
    
    return avg_psnr, avg_msssim


def main():
    parser = argparse.ArgumentParser(description='Train BK-SDM Compression Model')
    parser.add_argument('--bk_sdm_path', type=str, required=True, 
                       help='Path to BK-SDM model directory')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Training data path')
    parser.add_argument('--val_data_path', type=str, 
                       help='Validation data path')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size (reduced for memory constraints)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--lambda_rate', type=float, default=1e-4, 
                       help='Rate-distortion trade-off')
    parser.add_argument('--image_size', type=int, default=256, 
                       help='Image size')
    parser.add_argument('--N', type=int, default=128, 
                       help='Number of channels N')
    parser.add_argument('--M', type=int, default=128, 
                       help='Number of channels M')
    parser.add_argument('--diffusion_steps', type=int, default=30, 
                       help='Number of diffusion steps')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use')
    parser.add_argument('--resume', type=str, 
                       help='Resume from checkpoint')
    parser.add_argument('--use_pretrained_vae', action='store_true', 
                       help='Use pre-trained BK-SDM VAE')
    parser.add_argument('--use_pretrained_unet', action='store_true', 
                       help='Use pre-trained BK-SDM U-Net')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    setup_logging(output_dir)
    logging.info(f"Training on device: {device}")
    logging.info(f"Arguments: {vars(args)}")
    
    # Create model
    try:
        model = create_bk_sdm_compression_model(
            bk_sdm_path=args.bk_sdm_path,
            N=args.N,
            M=args.M,
            diffusion_steps=args.diffusion_steps,
            use_pretrained_vae=args.use_pretrained_vae,
            use_pretrained_unet=args.use_pretrained_unet
        ).to(device)
        
        logging.info("Model created successfully")
        
    except Exception as e:
        logging.error(f"Failed to create model: {e}")
        logging.info("Falling back to custom components")
        
        # Fallback to custom components
        model = create_bk_sdm_compression_model(
            bk_sdm_path=args.bk_sdm_path,
            N=args.N,
            M=args.M,
            diffusion_steps=args.diffusion_steps,
            use_pretrained_vae=False,
            use_pretrained_unet=False
        ).to(device)
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        args.data_path, args.batch_size, args.image_size
    )
    
    if args.val_data_path:
        val_dataloader = create_dataloader(
            args.val_data_path, args.batch_size, args.image_size
        )
    else:
        val_dataloader = None
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0
    
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint['best_psnr']
            logging.info(f"Resumed from checkpoint: {args.resume}")
    
    # Training loop
    logging.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_psnr, train_msssim = train_epoch(
            model, train_dataloader, optimizer, device, epoch, args.lambda_rate
        )
        
        # Validate
        val_psnr = val_msssim = 0
        if val_dataloader:
            val_psnr, val_msssim = validate(
                model, val_dataloader, device, output_dir / 'samples', epoch
            )
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        logging.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.6f}, "
            f"Train PSNR: {train_psnr:.2f}, "
            f"Train MS-SSIM: {train_msssim:.4f}"
        )
        
        if val_dataloader:
            logging.info(
                f"Epoch {epoch}: "
                f"Val PSNR: {val_psnr:.2f}, "
                f"Val MS-SSIM: {val_msssim:.4f}"
            )
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'args': vars(args)
        }
        
        torch.save(checkpoint, output_dir / 'latest_checkpoint.pth')
        
        # Save best model
        if val_dataloader and val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint['best_psnr'] = best_psnr
            torch.save(checkpoint, output_dir / 'best_checkpoint.pth')
            logging.info(f"New best PSNR: {best_psnr:.2f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    logging.info("Training completed!")
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pth')
    
    # Save training config
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    import torch.nn.functional as F
    main() 