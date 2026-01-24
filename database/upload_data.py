import os
import pickle
import uuid
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


############################################
# 1. LOAD GLOBAL MATCHING (LIST OF GROUPS)
############################################

def load_global_mapping_from_groups(pkl_path: str) -> Dict[Tuple[int, int, int], int]:
    """
    pkl format:
      [
        [(seq, cam, obj), ...],   # global_id = 0
        [(seq, cam, obj), ...],   # global_id = 1
        ...
      ]

    return:
      {(seq, cam, obj): global_id}
    """
    with open(pkl_path, "rb") as f:
        groups = pickle.load(f)

    mapping = {}
    for global_id, group in enumerate(groups):
        for key in group:
            mapping[tuple(key)] = global_id

    print(f"[INFO] Loaded {len(mapping)} track → global_id mappings")
    return mapping


############################################
# 2. LOAD METADATA TXT → DETECTIONS BY TRACK
############################################

def load_detections_by_track(txt_path: str):
    """
    txt line:
      seq_id, cam_id, obj_id, frame_id, x, y, w, h

    return:
      dict[(seq, cam, obj)] = [
        {"frame_id": int, "bbox": [x1, y1, x2, y2]},
        ...
      ]
    """
    tracks = defaultdict(list)

    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            seq_id, cam_id, frame_id, obj_id, x, y, x1, y1 = map(
                int, line.strip().split(" ")
            )

            tracks[(seq_id, cam_id, obj_id)].append({
                "frame_id": frame_id,
                "bbox": [x, y, x1, y1]
            })

    # sort by frame
    for k in tracks:
        tracks[k].sort(key=lambda d: d["frame_id"])

    return tracks


def load_all_detections(metadata_root: str):
    """
    metadata_root/
      seq_xxx/
        seq_xxx_camera_y.txt

    return:
      dict[(seq, cam, obj)] = detections[]
    """
    all_tracks = {}

    for seq_name in os.listdir(metadata_root):
        seq_dir = os.path.join(metadata_root, seq_name)
        if not os.path.isdir(seq_dir):
            continue

        for fname in os.listdir(seq_dir):
            if not fname.endswith(".txt"):
                continue

            txt_path = os.path.join(seq_dir, fname)
            tracks = load_detections_by_track(txt_path)

            for k, v in tracks.items():
                if k not in all_tracks:
                    all_tracks[k] = v
                else:
                    # safety (hiếm khi xảy ra)
                    all_tracks[k].extend(v)

    print(f"[INFO] Loaded detections for {len(all_tracks)} tracks")
    return all_tracks


############################################
# 3. LOAD FEATURE PKL (TRACK-LEVEL)
############################################

def load_feature_pkl(pkl_path: str):
    """
    expected format:
      dict[(seq, cam, obj)] = {
          "vector_reid": np.ndarray,
          "vector_clip": np.ndarray
      }
    """
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_all_features(feature_root: str):
    """
    feature_root/
      seq_xxx/
        seq_xxx_camera_y.pkl

    return:
      dict[(seq, cam, obj)] = feature
    """
    all_features = {}

    for seq_name in os.listdir(feature_root):
        seq_dir = os.path.join(feature_root, seq_name)
        if not os.path.isdir(seq_dir):
            continue

        for fname in os.listdir(seq_dir):
            if not fname.endswith(".pkl"):
                continue

            pkl_path = os.path.join(seq_dir, fname)
            feats = load_feature_pkl(pkl_path)

            for k, v in feats.items():
                if k in all_features:
                    print(f"[WARN] Duplicate feature for track {k}")
                all_features[k] = v

    print(f"[INFO] Loaded features for {len(all_features)} tracks")
    return all_features


############################################
# 4. BUILD QDRANT POINTS (1 TRACK = 1 POINT)
############################################

def build_qdrant_points(
    feature_dict,
    detections_dict,
    global_mapping
) -> List[PointStruct]:

    points = []
    skipped_no_det = 0

    for (seq_id, cam_id, obj_id), feat in feature_dict.items():

        if (seq_id, cam_id, obj_id) not in detections_dict:
            skipped_no_det += 1
            continue

        detections = detections_dict[(seq_id, cam_id, obj_id)]
        if len(detections) == 0:
            skipped_no_det += 1
            continue

        global_id = global_mapping.get((seq_id, cam_id, obj_id), None)

        point_id = str(uuid.uuid4())

        vectors = {
            "vector_reid": feat["reid"].tolist(),
            "vector_clip": feat["clip"].tolist()
        }

        payload = {
            "global_id": global_id,
            "seq_id": seq_id,
            "cam_id": cam_id,
            "obj_id": obj_id,
            "track_key": f"{seq_id}_{cam_id}_{obj_id}",
            "detections": detections,
            "frame_start": detections[0]["frame_id"],
            "frame_end": detections[-1]["frame_id"],
            "num_detections": len(detections)
        }

        points.append(PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload
        ))

    print(f"[INFO] Built {len(points)} points")
    print(f"[INFO] Skipped tracks without detections: {skipped_no_det}")
    return points


############################################
# 5. UPSERT TO QDRANT
############################################

def upsert_to_qdrant(
    points: List[PointStruct],
    collection_name="person_retrieval",
    host="localhost",
    port=6333,
    drop=False,
    batch_size=64
):
    client = QdrantClient(host=host, port=port)

    if drop and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"[INFO] Dropped collection {collection_name}")

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "vector_reid": VectorParams(
                    size=512,
                    distance=Distance.COSINE
                ),
                "vector_clip": VectorParams(
                    size=1024,
                    distance=Distance.COSINE
                )
            }
        )
        print(f"[INFO] Created collection {collection_name}")

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"[UPSERT] {i} -> {i + len(batch)}")

    print(f"[DONE] Total points upserted: {len(points)}")


############################################
# 6. MAIN PIPELINE (ALL SEQ, ALL CAM)
############################################

def main():
    # ===== PATH CONFIG =====
    FEATURE_ROOT = "data/new_feature_objects"
    METADATA_ROOT = "data/new_metadata_objects"
    GLOBAL_MATCH_PKL = "global_matching_results_2 (1).pkl"

    COLLECTION_NAME = "person_retrieval"

    print("========== START PIPELINE ==========")

    print("[STEP 1] Load global matching")
    global_mapping = load_global_mapping_from_groups(GLOBAL_MATCH_PKL)

    print("[STEP 2] Load all metadata")
    detections = load_all_detections(METADATA_ROOT)

    print("[STEP 3] Load all features")
    features = load_all_features(FEATURE_ROOT)

    print("[STEP 4] Build Qdrant points")
    points = build_qdrant_points(
        feature_dict=features,
        detections_dict=detections,
        global_mapping=global_mapping
    )

    print("[STEP 5] Upsert to Qdrant")
    upsert_to_qdrant(
        points,
        collection_name=COLLECTION_NAME,
        drop=True,
        batch_size=64
    )

    print("========== PIPELINE FINISHED ==========")


if __name__ == "__main__":
    main()
