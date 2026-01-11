import numpy as np


def collect_embeddings(tracklets):
    reid, clip = {}, {}

    for t in tracklets:
        if t.reid_embeddings:
            reid[t.global_id] =  t.reid_embeddings[0]

        if t.clip_embeddings:
            clip[t.global_id] = t.clip_embeddings[0]

    return reid, clip


def save_embeddings(reid, clip, out_dir="output"):
    np.save(f"{out_dir}/reid_embeddings.npy", reid)
    np.save(f"{out_dir}/clip_embeddings.npy", clip)

    print("[INFO] Saved embeddings (.npy)")
