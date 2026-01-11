import open_clip
import torch
from PIL import Image
import concurrent.futures
from typing import List, Tuple
import cv2
import torch.nn.functional as F
import numpy as np

class CLIPModel:
    def __init__(self, device='cpu'):
        ''' 

        '''
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14-quickgelu", pretrained="dfn5b", device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")
        self.device = device

    def preprocess_frame(self, frame, preprocess):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return preprocess(pil_image).unsqueeze(0)  # (1, C, H, W)

    @torch.no_grad()
    def encode_image(self, image):
        img = self.preprocess(Image.fromarray(image)).unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            return self.model.encode_image(img).cpu().numpy()[0]
    
    @torch.no_grad()
    def encode_batch(self, frames: List) -> np.ndarray:
        '''
        Docstring for encode_batch
        
        :param frames: Description
        :type frames: List
        :return: features of images (normalized numpy arrays)
        :rtype: numpy.ndarray
        '''
        # Parallelize PIL+preprocess on CPU
        with concurrent.futures.ThreadPoolExecutor() as ex:
            processed = list(ex.map(lambda fr: self.preprocess_frame(fr, self.preprocess), frames))
        images = torch.cat(processed, dim=0).to(self.device, non_blocking=True)  # (B, C, H, W)
        
        feats = self.model.encode_image(images)  # (B, D)
        feats = F.normalize(feats, dim=-1)  # unit-length for cosine similarity
        
        return feats.cpu().numpy()


    @torch.no_grad()
    def encode_text(self, text):
        tokens = self.tokenizer([text])
        tokens = tokens.to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens).cpu().numpy()[0]

