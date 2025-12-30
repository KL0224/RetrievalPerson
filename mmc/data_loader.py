import numpy as np
import re
import ast
from config import config

def load_metadata_file(file_path):
    # Giả sử mỗi dòng có định dạng: cam_id, frame_id, object_id, x, y, w, h
    metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7: continue
            cam_id, frame_id, object_id = int(parts[0]), int(parts[1]), int(parts[2])
            bbox = list(map(float, parts[3:7]))
            key = (cam_id, object_id)
            if key not in metadata:
                metadata[key] = []
            metadata[key].append({'frame_id': frame_id, 'bbox': bbox})
    return metadata

def load_features_file(file_path):
    # Giả sử mỗi dòng có định dạng: int, int, list, list (cam_id, object_id, vector_reid, vector_clip)
    features = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')

            # Vector
            lists = re.findall(r"\[.*?\]", line)
            if len(lists) != 2: continue

            vector_reid = ast.literal_eval(lists[0])
            vector_clip = ast.literal_eval(lists[1])

            if len(vector_reid) != config.reid_dim or len(vector_clip) != config.reid_dim:
                print(f"Lỗi: Chiều của vector reid hoặc clip không khớp!")
                continue

            # Cam + Object
            prefix = line.split("[")[0].strip()
            cam_id, object_id = map(int, prefix.split(",")[:2])

            key = (cam_id, object_id)
            features[key] = {
                "vector_reid": vector_reid,
                "vector_clip": vector_clip,
            }
    return features

def load_all_data(cam_files):
    # Dịnh dạng: cam_id: (meta_file_path, feature_file_path)
    all_metadatas = {}
    all_features = {}
    for cam_id, (meta_file, features_file) in cam_files.items():
        all_metadatas.update(load_metadata_file(meta_file))
        all_features.update(load_features_file(features_file))
    return all_metadatas, all_features

