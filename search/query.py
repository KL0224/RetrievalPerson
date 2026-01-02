import faiss
import numpy as np
import open_clip
import torch

from models.clip_model import CLIPModel


class FaissSearcher:
    def __init__(
        self,
        index_path: str,
        ids_path: str
    ):
        self.index = faiss.read_index(index_path)
        self.ids = np.load(ids_path)

    def search(self, query_embedding, top_k=10):
        q = query_embedding.astype("float32")
        q /= np.linalg.norm(q)

        scores, idxs = self.index.search(q[None, :], top_k)

        return [
            (int(self.ids[i]), float(scores[0][j]))
            for j, i in enumerate(idxs[0])
        ]

class ClipTextSearcher(FaissSearcher):
    def __init__(self, index_path, ids_path, clip_model: CLIPModel):
        super().__init__(index_path, ids_path)
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
        self.model = clip_model.model

    def encode_text(self, text):
        tokens = self.tokenizer([text]) #.cuda()
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        emb = emb.cpu().numpy()[0]
        return emb / np.linalg.norm(emb)

    def search_text(self, text, top_k=10):
        emb = self.encode_text(text)
        return self.search(emb, top_k)

