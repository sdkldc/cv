from compressai.models.utils import conv, deconv
import torch.nn as nn

"""
Compression pipeline integrating VAE, HyperEncoder, and LatentDiffusion.
"""

from .vae_stable_diffusion import VAEStableDiffusion
from .latent_diffusion import LatentDiffusion

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from .entropy import EntropyPred

class CompressionPipeline:
    """
    전체 인코딩/디코딩 파이프라인을 구현하는 클래스입니다.
    """
    def __init__(self, vae_path, diffusion_path, z_channels=192, y_channels=192):
        self.vae = VAEStableDiffusion(vae_path)
        self.latent_diffusion = LatentDiffusion(diffusion_path)
        self.entropy_bottleneck = EntropyBottleneck(channels=z_channels)  # z의 채널 수 지정
        self.gaussian_conditional = GaussianConditional(None)
        self.entropy_pred = EntropyPred(y_channels, y_channels)
        
        M = y_channels
        N = z_channels
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            # 마지막에는 conv하며 세부 조정
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.vae.encode(x)
        z = self.h_a.forward(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z) 
        y_pred = self.latent_diffusion.predict(self.h_s(z_hat))
        gaussian_params = self.entropy_pred(y_pred)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.vae.decode(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods} }

    def compress(self, x):
        """
        이미지를 입력받아 비트 스트림으로 저장
        """
        y = self.vae.encode(x)
        z = self.h_a.forward(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        y_pred = self.latent_diffusion.predict(self.h_s(z_hat))
        gaussian_params = self.entropy_pred(y_pred)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        """
        비트스트림(strings, shape)을 이미지로 복원
        """
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        y_pred = self.latent_diffusion.predict(self.h_s(z_hat))
        gaussian_params = self.entropy_pred(y_pred)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        x_hat = self.vae.decode(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
            

