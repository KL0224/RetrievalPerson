import open_clip
import torch
from PIL import Image

class CLIPModel:
    def __init__(self, device='cpu'):
        ''' 

        '''
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14-quickgelu", pretrained="dfn5b", device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14-quickgelu")

    def encode_image(self, image):
        img = self.preprocess(Image.fromarray(image)).unsqueeze(0)
        with torch.no_grad():
            return self.model.encode_image(img).cpu().numpy()[0]
    
    def encode_text(self, text):
        tokens = self.tokenizer([text])
        with torch.no_grad():
            return self.model.encode_text(tokens).cpu().numpy()[0]

