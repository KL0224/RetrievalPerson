import torch
import torchreid
from torchvision import transforms
from PIL import Image

class ReIDModel:
    def __init__(self, device='gpu'):
        self.model = torchreid.utils.feature_extractor.FeatureExtractor(
            model_name="osnet_x1_0",
            device=device,
            model_path='osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
        )
    def extract(self, images):
        feats = self.model(images)
        return feats.cpu().numpy()
