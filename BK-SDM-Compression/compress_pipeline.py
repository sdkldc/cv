import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.base import CompressionModel
from compressai.models.utils import conv
from compressai.layers import GDN


class CompressPipeline(CompressionModel):
    """
    CompressPipeline (새 버전)
    - 기존 Compression 파이프라인 구조 유지
    - VAE와 Latent Diffusion(UNet)은 BK-SDM(Diffusers) pre-trained 사용
    - y 압축은 GaussianConditional, z는 EntropyBottleneck 사용
    - scales_hat은 간단한 커스텀 UNet로 생성(구조 유지 목적)
    - 복원 시 UNet으로 latent 공간 denoise 후 VAE decode
    """

    def __init__(
        self,
        bk_sdm_path: str,
        N: int = 192,
        M: int = 192,
        diffusion_steps: int = 10,
        use_pretrained_vae: bool = True,
        use_pretrained_unet: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.N = int(N)
        self.M = int(M)
        self.diffusion_steps = int(diffusion_steps)
        self.bk_sdm_path = bk_sdm_path
        self.use_pretrained_vae = use_pretrained_vae
        self.use_pretrained_unet = use_pretrained_unet

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = torch.float16 if (self.device.type == "cuda") else torch.float32

        # Hyper Encoder (h_a): |y| -> z
        self.h_a = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
            nn.ReLU(inplace=True),
            conv(self.N, self.N),
        )

        # Entropy models
        self.entropy_bottleneck = EntropyBottleneck(self.N)
        self.gaussian_conditional = GaussianConditional(None)

        # means_hat & scales_hat 생성을 위한 소형 UNet (커스텀): 출력 채널 = 2*M (means | scales)
        self.scales_unet = _SmallScalesUNet(in_ch=self.M, hidden=self.N, out_ch=2 * self.M)
        self.scales_unet.to(self.device)

        # BK-SDM(Diffusers) 컴포넌트 로드
        self._load_bk_sdm_components()

        # GaussianConditional 기본 scale table
        try:
            scale_table = torch.exp(torch.linspace(-3.0, 10.0, 64))
            self.update(scale_table=scale_table, force=True)
        except Exception:
            pass

        self.to(self.device)

    @property
    def downsampling_factor(self) -> int:
        # Stable Diffusion VAE는 보통 8배 다운샘플
        return 8

    # ===== Public APIs =====
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1) VAE encode → latent
        y_latent = self._encode_with_vae(x)
        # 2) latent → 내부 M 채널
        y = self.y_to_M(y_latent).float()
        # 3) |y| → z
        z = self.h_a(torch.abs(y))
        # 4) Bottleneck
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # 5) z_hat → diffusion 입력으로 y_pred(latent) 생성
        y_pred_latent = self._y_from_diffusion_with_z(z_hat)
        y_pred_M = self.y_to_M(y_pred_latent).float()
        # 6) y_pred_M, z_hat → means_hat, scales_hat
        means_hat, scales_hat = self._generate_scales(y_pred_M, context=z_hat)
        # 7) 복원 경로(학습 패스에서는 원래 y를 decode하여 x_hat 계산)
        x_hat = self._decode_with_vae(self.M_to_latent(y))
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": None, "z": z_likelihoods},
            "y": y,
            "y_latent": y_latent,
            "z": z,
            "z_hat": z_hat,
            "y_pred_latent": y_pred_latent,
            "means_hat": means_hat,
            "scales_hat": scales_hat,
        }

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> Dict[str, any]:
        y_latent = self._encode_with_vae(x)
        y = self.y_to_M(y_latent).float()
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # z_hat → y_pred → means/scales
        y_pred_latent = self._y_from_diffusion_with_z(z_hat)
        y_pred_M = self.y_to_M(y_pred_latent).float()
        means_hat, scales_hat = self._generate_scales(y_pred_M, context=z_hat)

        # GaussianConditional: indexes(scales) + means 사용
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
        }

    @torch.no_grad()
    def decompress(self, strings: List[bytes], shape: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        assert isinstance(strings, list) and len(strings) == 2
        # z 복원
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        # z_hat → y_pred → means/scales 생성
        y_pred_latent = self._y_from_diffusion_with_z(z_hat)
        y_pred_M = self.y_to_M(y_pred_latent).float()
        means_hat, scales_hat = self._generate_scales(y_pred_M, context=z_hat)
        # y 복원 (indexes & means)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat_M = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype, means=means_hat)
        # latent 채널로 투영
        y_hat_latent = self.M_to_latent(y_hat_M.to(self.device, dtype=self.model_dtype))
        # 선택: 추가 latent denoise를 생략하고 바로 decode (요청 흐름에 맞춤)
        x_hat = self._decode_with_vae(y_hat_latent).clamp_(0, 1)
        return {"x_hat": x_hat}

    def update(self, scale_table=None, force=False):
        updated = self.entropy_bottleneck.update(force=force)
        updated |= self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def aux_loss(self) -> torch.Tensor:
        return self.entropy_bottleneck.loss()

    # ===== BK-SDM (Diffusers) components =====
    def _load_bk_sdm_components(self) -> None:
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        dtype = self.model_dtype
        print(f"Loading BK-SDM pipeline from: {self.bk_sdm_path}")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.bk_sdm_path,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(self.device)

        self.vae = pipe.vae.eval().to(self.device, dtype=dtype)
        self.unet = pipe.unet.eval().to(self.device, dtype=dtype)
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        # Latent 채널 수(보통 4)
        self.latent_channels = getattr(self.vae.config, "latent_channels", 4)
        # 채널 어댑터
        self.y_to_M = nn.Conv2d(self.latent_channels, self.M, kernel_size=1).to(self.device, dtype=dtype)
        self.M_to_latent = nn.Conv2d(self.M, self.latent_channels, kernel_size=1).to(self.device, dtype=dtype)
        # z_hat → latent 초기값 어댑터 (Diffusion 입력을 z_hat 기반으로 만들기 위함)
        self.z_to_latent = nn.Conv2d(self.N, self.latent_channels, kernel_size=1).to(self.device, dtype=dtype)

        # 무조건 임베딩(빈 프롬프트) 캐시
        tok = pipe.tokenizer([""], return_tensors="pt")
        with torch.no_grad():
            self.null_embeds = pipe.text_encoder(tok.input_ids.to(self.device))[0]

        # Freeze pre-trained
        for p in self.vae.parameters():
            p.requires_grad = False
        for p in self.unet.parameters():
            p.requires_grad = False

    # ===== Helper functions =====
    @torch.no_grad()
    def _encode_with_vae(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=self.model_dtype)
        x_scaled = x * 2.0 - 1.0
        posterior = self.vae.encode(x_scaled)
        latents = posterior.latent_dist.sample() * 0.18215
        return latents

    @torch.no_grad()
    def _decode_with_vae(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.device, dtype=self.model_dtype) / 0.18215
        x = self.vae.decode(latents).sample
        x = (x + 1.0) / 2.0
        return x

    @torch.no_grad()
    def _latent_denoise(self, latents: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        if num_steps <= 0:
            return latents
        x_t = latents.to(self.device, dtype=self.model_dtype)
        self.scheduler.set_timesteps(num_steps, device=self.device)
        for t in self.scheduler.timesteps:
            noise_pred = self.unet(sample=x_t, timestep=t, encoder_hidden_states=self.null_embeds).sample
            x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample
        return x_t

    @torch.no_grad()
    def _y_from_diffusion_with_z(self, z_hat: torch.Tensor) -> torch.Tensor:
        """z_hat을 diffusion의 입력으로 하여 y_pred(latent)를 생성."""
        init_latent = self.z_to_latent(z_hat.to(self.device, dtype=self.model_dtype))
        y_pred = self._latent_denoise(init_latent, num_steps=self.diffusion_steps)
        return y_pred

    @torch.no_grad()
    def _generate_scales(self, y_feature_M: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """y_feature_M와 z_hat(context)로부터 means_hat, scales_hat(M 채널)을 예측."""
        B, _, H, W = y_feature_M.shape
        tvec = self._get_timestep_embedding(
            torch.full((B,), max(self.diffusion_steps - 1, 0), device=y_feature_M.device, dtype=torch.long), self.N
        )
        out = self.scales_unet(y_feature_M.to(self.device, dtype=self.model_dtype), tvec, context=context)
        out = out.float()
        means_hat, scales_raw = torch.split(out, self.M, dim=1)
        scales_hat = F.softplus(scales_raw) + 1e-6
        return means_hat, scales_hat

    def _get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


class _SmallScalesUNet(nn.Module):
    """작은 UNet: scales_hat 생성용"""

    def __init__(self, in_ch: int, hidden: int, out_ch: int) -> None:
        super().__init__()
        self.time_fc = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden)
        )
        self.ctx_proj = nn.Conv2d(hidden, hidden, 1)

        self.enc1 = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.enc2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.dec1 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, out_ch, 3, padding=1)

        self.norm1 = nn.GroupNorm(16, hidden)
        self.norm2 = nn.GroupNorm(16, hidden)
        self.norm3 = nn.GroupNorm(16, hidden)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # t_embed: [B, hidden]
        B = x.shape[0]
        t = self.time_fc(t_embed)
        t = t.unsqueeze(-1).unsqueeze(-1)

        h = self.enc1(x)
        if context is not None:
            # context: [B,N,H,W] → proj to hidden
            h = h + self.ctx_proj(context)
        h = self.norm1(F.silu(h + t))
        h = self.norm2(F.silu(self.enc2(h) + t))
        h = self.norm3(F.silu(self.dec1(h) + t))
        return self.out(F.silu(h))


def create_compress_pipeline(
    bk_sdm_path: str,
    N: int = 192,
    M: int = 192,
    diffusion_steps: int = 10,
    use_pretrained_vae: bool = True,
    use_pretrained_unet: bool = True,
) -> CompressPipeline:
    return CompressPipeline(
        bk_sdm_path=bk_sdm_path,
        N=N,
        M=M,
        diffusion_steps=diffusion_steps,
        use_pretrained_vae=use_pretrained_vae,
        use_pretrained_unet=use_pretrained_unet,
    )


if __name__ == "__main__":
    # 간단한 동작 테스트
    path = "runwayml/stable-diffusion-v1-5"
    model = create_compress_pipeline(path, N=192, M=192, diffusion_steps=10)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    print(out["x_hat"].shape)


