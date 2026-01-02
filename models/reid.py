import torch
import torchreid
from torchvision import transforms
from PIL import Image

class ReIDModel:
    def __init__(self, device='cpu'):
        '''
        Docstring for __init__
        Defautl device is CPU
        :param self: Description
        :param device: Description
        '''
        self.model = torchreid.utils.feature_extractor.FeatureExtractor(
            model_name="osnet_x1_0",
            device=device,
            model_path='osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'
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
        :return: numpy array of extracted features
        '''
        feats = self.model(images)
        return feats.cpu().numpy()
        # return self.feature_extractor(images)
        #     feats = self.model(imgs)
        # return feats.cpu().numpy()
