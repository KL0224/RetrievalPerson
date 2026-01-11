import faiss
import numpy as np
from typing import Dict


def build_faiss_index(
    embeddings: Dict[int, np.ndarray],
    index_path: str,
    ids_path: str
):
    """
    embeddings: {tracklet_id: embedding}
    """

    tracklet_ids = np.array(list(embeddings.keys()), dtype=np.int64)
    vectors = np.stack(list(embeddings.values())).astype("float32")

    # L2 normalize (cosine similarity)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, index_path)
    np.save(ids_path, tracklet_ids)

    print(f"[FAISS] Built index: {index.ntotal} vectors -> {index_path}")
