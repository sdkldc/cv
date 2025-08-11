import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.base import CompressionModel
from compressai.layers import GDN
from compressai.models.utils import conv, deconv


class DiffusionHyperpriorCompression(CompressionModel):
    """
    Diffusion-based Hyperprior Compression Model
    
    Architecture:
    1. VAE Encoder (g_a): Input image -> Latent space (y)
    2. Hyper Encoder (h_a): |y| -> z (hyperprior)
    3. Entropy Bottleneck: z -> z_hat (compressed)
    4. Hyper Decoder (h_s): z_hat -> scales_hat
    5. Diffusion Denoising: z_hat -> y_hat (denoised latent)
    6. VAE Decoder (g_s): y_hat -> x_hat (reconstructed image)
    
    The diffusion process helps in better entropy estimation and reconstruction quality.
    """
    
    def __init__(self, N=192, M=192, diffusion_steps=100, **kwargs):
        super().__init__(**kwargs)
        
        self.N = int(N)
        self.M = int(M)
        self.diffusion_steps = diffusion_steps
        
        # VAE Encoder (g_a): Input -> Latent
        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )
        
        # VAE Decoder (g_s): Latent -> Output
        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )
        
        # Hyper Encoder (h_a): |y| -> z
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )
        
        # Hyper Decoder (h_s): z_hat -> scales_hat
        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        
        # Diffusion UNet for denoising z_hat -> y_hat
        self.diffusion_unet = DiffusionUNet(
            in_channels=N,
            out_channels=M,
            model_channels=N,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0.1,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            num_heads=8,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=N,
            use_linear_projection=True,
            class_embed_type=None,
            num_class_embeds=None,
            upcast_attention=False,
            resnet_skip_time_act=False,
            resnet_out_scale_factor=1.0,
            resnet_time_scale_shift="default",
            resnet_cond_scale_shift="default",
            resnet_pre_temb_non_linearity=False,
        )
        
        # Entropy models
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        
        # Diffusion noise scheduler
        self.noise_scheduler = DiffusionNoiseScheduler(
            num_train_timesteps=diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            schedule="linear"
        )
        
        # Time embedding for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(N, N * 4),
            nn.SiLU(),
            nn.Linear(N * 4, N),
        )
        
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        """
        # VAE encoding
        y = self.g_a(x)
        
        # Hyper encoding
        z = self.h_a(torch.abs(y))
        
        # Entropy bottleneck
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        # Hyper decoding to get scales
        scales_hat = self.h_s(z_hat)
        
        # Diffusion denoising process
        y_hat = self._diffusion_denoise(z_hat, scales_hat)
        
        # VAE decoding
        x_hat = self.g_s(y_hat)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": None, "z": z_likelihoods},  # y likelihoods will be computed separately
            "y": y,
            "y_hat": y_hat,
            "z": z,
            "z_hat": z_hat,
            "scales_hat": scales_hat,
        }
    
    def _diffusion_denoise(self, z_hat: torch.Tensor, scales_hat: torch.Tensor) -> torch.Tensor:
        """
        Diffusion denoising process: z_hat -> y_hat
        """
        batch_size = z_hat.shape[0]
        device = z_hat.device
        
        # Start from pure noise
        y_t = torch.randn_like(z_hat, device=device)
        
        # Reverse diffusion process
        for t in range(self.diffusion_steps - 1, -1, -1):
            # Create timestep tensor
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get time embedding
            time_emb = self.time_embed(self._get_timestep_embedding(timesteps, self.N))
            
            # Predict noise
            noise_pred = self.diffusion_unet(
                y_t,
                timesteps,
                context=z_hat,
                scales=scales_hat,
                time_emb=time_emb
            )
            
            # Denoising step
            alpha_t = self.noise_scheduler.alphas[t]
            alpha_t_prev = self.noise_scheduler.alphas[t-1] if t > 0 else torch.tensor(1.0)
            
            # DDPM denoising formula
            y_t = (1 / torch.sqrt(alpha_t)) * (
                y_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_t)) * noise_pred
            )
            
            if t > 0:
                noise = torch.randn_like(y_t, device=device)
                y_t = y_t + torch.sqrt(1 - alpha_t_prev) * noise
        
        return y_t
    
    def _get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb
    
    def compress(self, x: torch.Tensor) -> Dict[str, any]:
        """
        Compress input image
        """
        # VAE encoding
        y = self.g_a(x)
        
        # Hyper encoding
        z = self.h_a(torch.abs(y))
        
        # Compress z
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        # Get scales for diffusion
        scales_hat = self.h_s(z_hat)
        
        # Compress y using Gaussian conditional
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "z_hat": z_hat,
            "scales_hat": scales_hat
        }
    
    def decompress(self, strings: List[bytes], shape: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Decompress from compressed strings
        """
        assert isinstance(strings, list) and len(strings) == 2
        
        # Decompress z
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        
        # Get scales
        scales_hat = self.h_s(z_hat)
        
        # Decompress y
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        
        # Apply diffusion denoising
        y_hat_denoised = self._diffusion_denoise(z_hat, scales_hat)
        
        # VAE decoding
        x_hat = self.g_s(y_hat_denoised).clamp_(0, 1)
        
        return {"x_hat": x_hat}
    
    def update(self, scale_table=None, force=False):
        """
        Update entropy model parameters
        """
        updated = self.entropy_bottleneck.update(force=force)
        updated |= self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated
    
    def aux_loss(self) -> torch.Tensor:
        """
        Auxiliary loss for entropy models
        """
        return self.entropy_bottleneck.loss()


class DiffusionUNet(nn.Module):
    """
    Simplified UNet for diffusion denoising
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: Tuple[int, ...],
        dropout: float = 0.0,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_heads: int = 8,
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        resnet_cond_scale_shift: str = "default",
        resnet_pre_temb_non_linearity: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])
        
        # Downsampling blocks
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        resnet_skip_time_act=resnet_skip_time_act,
                        resnet_out_scale_factor=resnet_out_scale_factor,
                        resnet_cond_scale_shift=resnet_cond_scale_shift,
                        resnet_pre_temb_non_linearity=resnet_pre_temb_non_linearity,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=None,
                            use_new_attention_order=False,
                        )
                    )
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([Downsample(ch, conv_resample, dims=2)])
                )
                input_block_chans.append(ch)
                ds *= 2
        
        # Middle block
        self.middle_block = nn.ModuleList([
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                resnet_cond_scale_shift=resnet_cond_scale_shift,
                resnet_pre_temb_non_linearity=resnet_pre_temb_non_linearity,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=None,
                use_new_attention_order=False,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                resnet_cond_scale_shift=resnet_cond_scale_shift,
                resnet_pre_temb_non_linearity=resnet_pre_temb_non_linearity,
            ),
        ])
        
        # Upsampling blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        resnet_skip_time_act=resnet_skip_time_act,
                        resnet_out_scale_factor=resnet_out_scale_factor,
                        resnet_cond_scale_shift=resnet_cond_scale_shift,
                        resnet_pre_temb_non_linearity=resnet_pre_temb_non_linearity,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=None,
                            use_new_attention_order=False,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=2))
                    ds //= 2
                self.output_blocks.append(nn.ModuleList(layers))
        
        # Output projection
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, kernel_size=3, padding=1),
        )
        
        # Context projection (for z_hat)
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, model_channels)
        else:
            self.context_proj = None
        
        # Scales projection (for scales_hat)
        self.scales_proj = nn.Conv2d(model_channels, model_channels, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        """
        if time_emb is None:
            time_emb = self.time_embed(timesteps)
        
        # Context embedding (z_hat)
        if context is not None and self.context_proj is not None:
            context_emb = self.context_proj(context.mean(dim=[-2, -1]))  # Global average pooling
            time_emb = time_emb + context_emb.unsqueeze(1)
        
        # Scales embedding
        if scales is not None:
            scales_emb = self.scales_proj(scales)
            x = x + scales_emb
        
        # Input blocks
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, time_emb)
                    else:
                        h = layer(h)
                hs.append(h)
            else:
                h = module(h)
                hs.append(h)
        
        # Middle block
        for module in self.middle_block:
            if isinstance(module, ResBlock):
                h = module(h, time_emb)
            else:
                h = module(h)
        
        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, time_emb)
                else:
                    h = layer(h)
        
        return self.out(h)


class ResBlock(nn.Module):
    """
    Residual block with time embedding
    """
    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        dropout: float,
        out_channels: Optional[int] = None,
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        resnet_cond_scale_shift: str = "default",
        resnet_pre_temb_non_linearity: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.out_channels = out_channels or channels
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        
        self.time_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.out_channels),
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        time_emb = self.time_emb_layers(time_emb)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    Self-attention block
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: Optional[int] = None,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        
        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        scale = 1 / math.sqrt(math.sqrt(C))
        
        attn = torch.einsum("bchw,bcij->bhwij", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        
        h = torch.einsum("bhwij,bcij->bchw", attn, v)
        h = self.proj(h)
        
        return x + h


class Upsample(nn.Module):
    """
    Upsampling block
    """
    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.dims = dims
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 2:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        
        if self.use_conv:
            x = self.conv(x)
        
        return x


class Downsample(nn.Module):
    """
    Downsampling block
    """
    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.dims = dims
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.channels
        if self.use_conv:
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, 2)
        return x


class DiffusionNoiseScheduler:
    """
    Simple noise scheduler for diffusion process
    """
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear"
    ):
        self.num_train_timesteps = num_train_timesteps
        
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Pre-compute values for inference
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps: int) -> torch.Tensor:
        """
        Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


def create_diffusion_hyperprior_model(
    N: int = 192,
    M: int = 192,
    diffusion_steps: int = 100
) -> DiffusionHyperpriorCompression:
    """
    Factory function to create a diffusion hyperprior compression model
    """
    return DiffusionHyperpriorCompression(
        N=N,
        M=M,
        diffusion_steps=diffusion_steps
    )


if __name__ == "__main__":
    # Example usage
    model = create_diffusion_hyperprior_model(N=128, M=128, diffusion_steps=50)
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output['x_hat'].shape}")
    print(f"Latent shape: {output['y'].shape}")
    print(f"Hyperprior shape: {output['z'].shape}")
    
    # Test compression/decompression
    compressed = model.compress(x)
    decompressed = model.decompress(compressed['strings'], compressed['shape'])
    
    print(f"Compression successful: {len(compressed['strings'])} strings")
    print(f"Decompression successful: {decompressed['x_hat'].shape}") 