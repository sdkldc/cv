import torch
import torch.nn as nn

class EntropyPred(nn.Module):
	"""
	Latent space에서 가우시안 분포의 파라미터(스케일, 평균 등)를 예측하는 작은 CNN 네트워크.
	입력: latent feature (y_pred)
	출력: [scales, means] (채널 2배)
	"""
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, out_channels * 2, kernel_size=3, padding=1)  # scale, mean
		)

	def forward(self, x):
		return self.net(x)
