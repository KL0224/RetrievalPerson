import torch
import torchreid
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class ReIDModel:
    def __init__(self, device='cpu', model_path=None):
        '''
        Docstring for __init__
        Defautl device is CPU
        :param self: Description
        :param device: Description
        '''
        self.model = torchreid.utils.feature_extractor.FeatureExtractor(
            model_name="osnet_x1_0",
            device=device,
            model_path=model_path
        )
        # self.model = torchreid.models.build_model(
        #     name="osnet_x1_0",
        #     num_classes=1000,
        #     pretrained=True
        # ).eval().cuda()

        # self.tf = transforms.Compose([
        #     transforms.Resize((256, 128)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ])

    def extract(self, images):
        '''
        Docstring for extract
        :param images: can be a list of images in numpy array format, list of paths, list of tensors
        :return: NORMALIZED numpy array of extracted features
        '''
        feats = self.model(images)
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy()
        