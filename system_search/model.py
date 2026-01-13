import open_clip
import torch
from PIL import Image
import torchreid
from PIL import Image
import torch.nn.functional as F

class CLIPModel:
    def __init__(self, device='cuda'):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14-quickgelu", pretrained="dfn5b", device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
        self.device = device

    def encode_image(self, image):
        img = self.preprocess(Image.fromarray(image)).unsqueeze(0)
        with torch.no_grad():
            feat = self.model.encode_image(img.to(self.device))
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy()[0]

    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat = F.normalize(feat, dim=-1)
        return feat.cpu().numpy()[0]

class ReIDModel:
    def __init__(self, device='cuda', model_path=None):
        self.model = torchreid.utils.feature_extractor.FeatureExtractor(
            model_name="osnet_x1_0",
            device=device,
            model_path='models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth' if model_path is None else model_path
        )
        self.device = device

    def extract(self, images):
        with torch.no_grad():
            feats = self.model(images)
            feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy()



