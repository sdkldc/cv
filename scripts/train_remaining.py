import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


def add_repo_module_path():
    repo_root = Path(__file__).resolve().parents[1]
    module_dir = repo_root / "BK-SDM-Compression"
    if str(module_dir) not in os.sys.path:
        os.sys.path.append(str(module_dir))


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int = 256):
        self.root_dir = root_dir
        self.files = [
            f for f in os.listdir(root_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.files.sort()
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x


def train_one_epoch(model, loader, optimizer, device, lmbda: float = 1.0):
    model.train()
    total_loss, total_rec, total_aux = 0.0, 0.0, 0.0

    for x in loader:
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        x_hat = out["x_hat"].clamp(0, 1)
        rec = F.mse_loss(x_hat, x)
        aux = model.aux_loss()
        loss = lmbda * rec + 1e-3 * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_rec += rec.item() * x.size(0)
        total_aux += aux.item() * x.size(0)

    n = len(loader.dataset)
    return total_loss / n, total_rec / n, total_aux / n


def main():
    parser = argparse.ArgumentParser(
        description="Train remaining modules only (VAE/UNet frozen)"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bk_sdm_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=192)
    parser.add_argument("--diffusion_steps", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--use_pipeline", action="store_true", help="Use CompressPipeline instead of BKSDMCompression")
    args = parser.parse_args()

    add_repo_module_path()
    # CompressPipeline
    from compress_pipeline import create_compress_pipeline as create_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Build model (VAE/UNet are frozen inside)
    model = create_model(
        bk_sdm_path=args.bk_sdm_path,
        N=args.N,
        M=args.M,
        diffusion_steps=args.diffusion_steps,
        use_pretrained_vae=True,
        use_pretrained_unet=True,
    )
    model.to(device)

    # Trainable params only
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.Adam(trainable, lr=args.lr)

    # Data
    ds = ImageFolderDataset(args.data_dir, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    for epoch in range(1, args.epochs + 1):
        loss, rec, aux = train_one_epoch(model, dl, optimizer, device)
        # Update entropy modules
        model.update(force=True)
        ckpt_path = os.path.join(args.save_dir, f"rem-only-epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": {"loss": loss, "rec": rec, "aux": aux},
        }, ckpt_path)
        print(f"Epoch {epoch}: loss={loss:.6f} rec={rec:.6f} aux={aux:.6f} → saved {ckpt_path}")


if __name__ == "__main__":
    main()


