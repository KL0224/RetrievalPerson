import numpy as np

from config import *
from tracking.tracklet import TrackletManager
from tracking.detector_tracker import run_tracking
from sampling.sampler import sample_best_per_window, group_seconds
from models.reid import ReIDModel
from models.clip_model import CLIPModel
from storage.database import init_db
from storage.embeddings import collect_embeddings, save_embeddings
from storage.faiss_index import build_faiss_index


manager = TrackletManager()
reid_model = ReIDModel()
clip_model = CLIPModel()
conn = init_db("output/tracklets.db")
cur = conn.cursor()



for cam_id, video in CAMERAS.items():
    for frame_id, frame, boxes, ids, confs in run_tracking(video, vid_stride=VID_STRIDE, confidence=CONFIDENCE_THRESHOLD):
        for box, tid, conf in zip(boxes, ids, confs):
            gid = cam_id * GLOBAL_ID_OFFSET + tid
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]

            t = manager.get(gid, cam_id)
            t.add_frame(frame_id, box, conf, crop)

            cur.execute(
                "INSERT INTO frame VALUES (NULL,?,?,?,?,?,?,?)",
                (gid, frame_id, x1, y1, x2, y2, conf)
            )

conn.commit()

# Sampling + Embeddings
for t in manager.all():
    sampled = sample_best_per_window(t.frames)
    candidates = []
    if len(sampled) <= 3:
        candidates = sampled
    else:
        candidates = sampled[:1] + sampled[-1:] + sampled[len(sampled)//2:len(sampled)//2+1]

    imgs = [f.image for f in candidates]

    reid_feats = reid_model.extract(imgs).mean(axis=0)
    t.reid_embeddings.append(reid_feats)

    clip_feats = np.array([clip_model.encode_image(img) for img in imgs]).mean(axis=0)
    t.clip_embeddings.append(clip_feats)
    # groups = group_seconds(sampled)

    # for g in groups:
    #     # 1 group: 3 frames
    #     imgs = [f.image for f in g]
    #     reid_feats = reid_model.extract(imgs)
    #     t.reid_embeddings.append(reid_feats.mean(axis=0))
    #     t.clip_embeddings.append(clip_model.encode_image(imgs[0]))

# ---- AFTER sampling + embedding extraction ----

tracklets = manager.all()

# save into features.txt
with open("output/features.txt", "w") as f:
    for t in tracklets:
        f.write(f"{t.camera_id} {t.global_id} {t.reid_embeddings[0].tolist()} {t.clip_embeddings[0].tolist()}\n")

# Collect embeddings
reid_embs, clip_embs = collect_embeddings(tracklets)

# Save embeddings
save_embeddings(reid_embs, clip_embs)

# ---- Build FAISS indexes ----
# for REID
build_faiss_index(
    reid_embs,
    "output/faiss_reid.index",
    "output/faiss_reid_ids.npy"
)

# for CLIP
build_faiss_index(
    clip_embs,
    "output/faiss_clip.index",
    "output/faiss_clip_ids.npy"
)
