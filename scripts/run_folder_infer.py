import os
import sys
import math
import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image


def add_repo_module_path():
    repo_root = Path(__file__).resolve().parents[1]
    module_dir = repo_root / "BK-SDM-Compression"
    sys.path.append(str(module_dir))


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def count_bits(strings_obj) -> int:
    total_bytes = 0
    for group in strings_obj:
        if isinstance(group, (list, tuple)):
            for s in group:
                total_bytes += len(s)
        elif isinstance(group, (bytes, bytearray)):
            total_bytes += len(group)
    return total_bytes * 8


def main():
    parser = argparse.ArgumentParser(description="Folder inference for BK-SDM Compression (no retraining)")
    parser.add_argument("--data_dir", type=str, required=True, help="Input image folder path")
    parser.add_argument("--bk_sdm_path", type=str, default="", help="Diffusers BK-SDM/SD pipeline path or HF model id")
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=192)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--use_pretrained_vae", action="store_true", help="Use pre-trained VAE from BK-SDM")
    parser.add_argument("--use_pretrained_unet", action="store_true", help="Use pre-trained UNet from BK-SDM")
    parser.add_argument("--save_dir", type=str, default="", help="Optional folder to save reconstructed images")
    args = parser.parse_args()

    add_repo_module_path()
    from compress_pipeline import create_compress_pipeline

    model = create_compress_pipeline(
        bk_sdm_path=args.bk_sdm_path,
        N=args.N,
        M=args.M,
        diffusion_steps=args.diffusion_steps,
        use_pretrained_vae=args.use_pretrained_vae,
        use_pretrained_unet=args.use_pretrained_unet,
    )
    model.eval()

    to_tensor = T.Compose([T.ToTensor()])  # [0,1]

    files = [f for f in os.listdir(args.data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    with torch.no_grad():
        for fname in files:
            in_path = os.path.join(args.data_dir, fname)
            img = Image.open(in_path).convert("RGB")
            x = to_tensor(img).unsqueeze(0)  # [1,3,H,W]

            comp = model.compress(x)
            decomp = model.decompress(comp["strings"], comp["shape"]) 
            x_hat = decomp["x_hat"].clamp(0, 1)

            H, W = x.shape[-2:]
            bpp = count_bits(comp["strings"]) / (H * W)
            print(f"{fname} | bpp: {bpp:.4f} | PSNR: {psnr(x, x_hat):.2f} dB")

            if args.save_dir:
                out_img = (x_hat.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype("uint8")
                Image.fromarray(out_img).save(os.path.join(args.save_dir, fname))


if __name__ == "__main__":
    main()


