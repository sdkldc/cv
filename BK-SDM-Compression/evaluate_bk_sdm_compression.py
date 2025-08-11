import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import logging
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
import time
import psutil
import gc
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
            logging.FileHandler(os.path.join(log_dir, 'evaluation.log')),
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
        shuffle=False,
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


def compute_compression_ratio(original_size: int, compressed_size: int) -> float:
    """압축률 계산"""
    return original_size / compressed_size


def get_memory_usage() -> dict:
    """메모리 사용량 확인"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024,  # MB
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str,
    save_samples: bool = True
):
    """모델 평가"""
    model.eval()
    
    total_psnr = 0
    total_msssim = 0
    total_compression_time = 0
    total_decompression_time = 0
    total_compression_ratio = 0
    total_original_size = 0
    total_compressed_size = 0
    
    results = []
    
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc='Evaluating')):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Original size (bytes)
            original_size = data.numel() * data.element_size()
            total_original_size += original_size
            
            # Compression
            start_time = time.time()
            compressed = model.compress(data)
            compression_time = time.time() - start_time
            total_compression_time += compression_time
            
            # Calculate compressed size
            compressed_size = sum(len(s) for s in compressed['strings'])
            total_compressed_size += compressed_size
            
            # Compression ratio
            compression_ratio = compute_compression_ratio(original_size, compressed_size)
            total_compression_ratio += compression_ratio
            
            # Decompression
            start_time = time.time()
            decompressed = model.decompress(compressed['strings'], compressed['shape'])
            decompression_time = time.time() - start_time
            total_decompression_time += decompression_time
            
            # Metrics
            psnr = compute_psnr(data, decompressed['x_hat'])
            msssim = compute_msssim(data, decompressed['x_hat'])
            
            total_psnr += psnr
            total_msssim += msssim
            
            # Save results
            batch_results = {
                'batch_idx': batch_idx,
                'psnr': psnr,
                'msssim': msssim,
                'compression_time': compression_time,
                'decompression_time': decompression_time,
                'compression_ratio': compression_ratio,
                'original_size': original_size,
                'compressed_size': compressed_size,
            }
            results.append(batch_results)
            
            # Save sample images
            if save_samples and batch_idx < 5:
                save_image(
                    torch.cat([data[:4], decompressed['x_hat'][:4]], dim=0),
                    os.path.join(save_dir, f'sample_batch_{batch_idx}.png'),
                    nrow=4,
                    normalize=True
                )
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # Calculate averages
    num_batches = len(dataloader)
    avg_psnr = total_psnr / num_batches
    avg_msssim = total_msssim / num_batches
    avg_compression_time = total_compression_time / num_batches
    avg_decompression_time = total_decompression_time / num_batches
    avg_compression_ratio = total_compression_ratio / num_batches
    
    # Overall compression ratio
    overall_compression_ratio = total_original_size / total_compressed_size
    
    # Memory usage
    memory_usage = get_memory_usage()
    
    evaluation_results = {
        'metrics': {
            'avg_psnr': avg_psnr,
            'avg_msssim': avg_msssim,
            'avg_compression_time': avg_compression_time,
            'avg_decompression_time': avg_decompression_time,
            'avg_compression_ratio': avg_compression_ratio,
            'overall_compression_ratio': overall_compression_ratio,
        },
        'totals': {
            'total_original_size': total_original_size,
            'total_compressed_size': total_compressed_size,
            'total_compression_time': total_compression_time,
            'total_decompression_time': total_decompression_time,
        },
        'memory_usage': memory_usage,
        'batch_results': results,
    }
    
    return evaluation_results


def save_evaluation_results(results: dict, save_path: str):
    """평가 결과 저장"""
    # Save detailed results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary
    summary_path = save_path.replace('.json', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== BK-SDM Compression Model Evaluation Summary ===\n\n")
        
        metrics = results['metrics']
        f.write(f"Average PSNR: {metrics['avg_psnr']:.2f} dB\n")
        f.write(f"Average MS-SSIM: {metrics['avg_msssim']:.4f}\n")
        f.write(f"Average Compression Time: {metrics['avg_compression_time']:.4f} seconds\n")
        f.write(f"Average Decompression Time: {metrics['avg_decompression_time']:.4f} seconds\n")
        f.write(f"Average Compression Ratio: {metrics['avg_compression_ratio']:.2f}x\n")
        f.write(f"Overall Compression Ratio: {metrics['overall_compression_ratio']:.2f}x\n\n")
        
        totals = results['totals']
        f.write(f"Total Original Size: {totals['total_original_size'] / 1024 / 1024:.2f} MB\n")
        f.write(f"Total Compressed Size: {totals['total_compressed_size'] / 1024 / 1024:.2f} MB\n")
        f.write(f"Total Compression Time: {totals['total_compression_time']:.2f} seconds\n")
        f.write(f"Total Decompression Time: {totals['total_decompression_time']:.2f} seconds\n\n")
        
        memory = results['memory_usage']
        f.write(f"Memory Usage:\n")
        f.write(f"  RSS: {memory['rss']:.2f} MB\n")
        f.write(f"  VMS: {memory['vms']:.2f} MB\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate BK-SDM Compression Model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--bk_sdm_path', type=str, required=True, 
                       help='Path to BK-SDM model directory')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Test data path')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', 
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=2, 
                       help='Batch size for evaluation (reduced for memory)')
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
    parser.add_argument('--save_samples', action='store_true', 
                       help='Save sample images')
    parser.add_argument('--config_path', type=str, 
                       help='Path to model config file')
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
    logging.info(f"Evaluation on device: {device}")
    logging.info(f"Arguments: {vars(args)}")
    
    # Load model configuration if provided
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded config: {config}")
    else:
        config = {
            'N': args.N,
            'M': args.M,
            'diffusion_steps': args.diffusion_steps
        }
    
    # Create model
    try:
        model = create_bk_sdm_compression_model(
            bk_sdm_path=args.bk_sdm_path,
            N=config.get('N', args.N),
            M=config.get('M', args.M),
            diffusion_steps=config.get('diffusion_steps', args.diffusion_steps),
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
            N=config.get('N', args.N),
            M=config.get('M', args.M),
            diffusion_steps=config.get('diffusion_steps', args.diffusion_steps),
            use_pretrained_vae=False,
            use_pretrained_unet=False
        ).to(device)
    
    # Load trained weights
    if os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        logging.info(f"Loaded model from: {args.model_path}")
    else:
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Create dataloader
    dataloader = create_dataloader(
        args.data_path, args.batch_size, args.image_size
    )
    
    logging.info(f"Dataset size: {len(dataloader.dataset)}")
    logging.info(f"Number of batches: {len(dataloader)}")
    
    # Evaluate model
    logging.info("Starting evaluation...")
    start_time = time.time()
    
    evaluation_results = evaluate_model(
        model, dataloader, device, output_dir / 'samples', args.save_samples
    )
    
    total_evaluation_time = time.time() - start_time
    
    # Add evaluation time to results
    evaluation_results['evaluation_time'] = total_evaluation_time
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    save_evaluation_results(evaluation_results, str(results_path))
    
    # Print summary
    metrics = evaluation_results['metrics']
    logging.info("=== Evaluation Summary ===")
    logging.info(f"Average PSNR: {metrics['avg_psnr']:.2f} dB")
    logging.info(f"Average MS-SSIM: {metrics['avg_msssim']:.4f}")
    logging.info(f"Overall Compression Ratio: {metrics['overall_compression_ratio']:.2f}x")
    logging.info(f"Total Evaluation Time: {total_evaluation_time:.2f} seconds")
    
    logging.info(f"Results saved to: {results_path}")
    logging.info("Evaluation completed!")


if __name__ == '__main__':
    main() 