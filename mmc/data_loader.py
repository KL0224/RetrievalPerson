import re
import ast
from mot.config import config

def load_metadata_file(file_path):
    # Giả sử mỗi dòng có định dạng: seq_id cam_id frame_id object_id x y w h
    metadata = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 8: continue
            seq_id, cam_id, frame_id, object_id = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            bbox = list(map(float, parts[4:8]))
            key = (seq_id, cam_id, object_id)
            if key not in metadata:
                metadata[key] = []
            metadata[key].append({'frame_id': frame_id, 'bbox': bbox})
    return metadata

def load_features_file(file_path):
    # Giả sử mỗi dòng có định dạng: int int int list list (seq_id, cam_id, object_id, vector_reid, vector_clip)
    features = {}
    with open(file_path, 'r') as f:
        for line in f:
            numbers = re.findall(r'\b\d+\b', line)

            # Vector
            lists = re.findall(r'\[[^\]]*\]', line)
            if len(lists) != 2: continue

            vector_reid = ast.literal_eval(lists[0])
            vector_clip = ast.literal_eval(lists[1])

            if len(vector_reid) != config.reid_dim or len(vector_clip) != config.reid_dim:
                print(f"Lỗi: Chiều của vector reid hoặc clip không khớp!")
                continue

            # Seq + Cam + Object
            seq_id, cam_id, object_id = numbers[0], numbers[1], numbers[2]

            key = (seq_id, cam_id, object_id)
            features[key] = {
                "vector_reid": vector_reid,
                "vector_clip": vector_clip,
            }
    return features

def load_all_data(cam_files):
    # Dịnh dạng: (meta_file_path, feature_file_path)
    all_metadatas = {}
    all_features = {}
    for meta_file, features_file in cam_files:
        all_metadatas.update(load_metadata_file(meta_file))
        all_features.update(load_features_file(features_file))
    return all_metadatas, all_features

