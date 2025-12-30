import numpy as np
from sklearn.cluster import OPTICS
from config import config
import networkx as nx

def compute_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def build_graph(objects, features, metadata):
    G = nx.Graph()
    obj_keys = list(objects)
    for i in range(len(obj_keys)):
        cam1, obj1 = obj_keys[i]
        vec1 = np.concatenate([features[(cam1, obj1)]['vector_reid'], features[(cam1, obj1)]['vector_clip']])
        for j in range(i + 1, len(obj_keys)):
            cam2, obj2 = obj_keys[j]
            if cam1 == cam2: continue
            vec2 = np.concatenate([features[(cam2, obj2)]['vector_reid'], features[(cam2, obj2)]['vector_clip']])
            sim = compute_similarity(vec1, vec2)
            if sim > config.similarity_threshold and cam2 in config.spatial_neighbors.get(cam1, []):
                G.add_edge((cam1, obj1), (cam2, obj2), weight=sim)
    return G

def perform_matching(G, features, metadata):
    components = list(nx.connected_components(G))
    global_objects = {}
    for idx, component in enumerate(components):
        global_id = f"global_{idx}"

        appearances = []
        detections = []
        for node in component:
            cam_id, obj_id = node

            # Lưu vector gốc của từng node (đại diện cho 1 lần xuất hiện)
            appearances.append({
                'vector_reid': features[node]['vector_reid'],
                'vector_clip': features[node]['vector_clip'],
                'cam_id': cam_id,
                'obj_id': obj_id
            })

            for det in metadata[node]:
                detections.append({
                    'cam_id': cam_id,
                    'frame_id': det['frame_id'],
                    'bbox': det['bbox'],
                    'timestamp': det['frame_id'] / config.fps
                })

        global_objects[global_id] = {
            'appearances': appearances,
            'avg_vector_reid': np.mean([a['vector_reid'] for a in appearances], axis=0),
            'avg_vector_clip': np.mean([a['vector_clip'] for a in appearances], axis=0),
            'metadata': {'detections': detections}
        }

    return global_objects

def apply_open_set_clustering(unmatched_objects, features, metadata):
    if not unmatched_objects:
        return {}

    vecs = [np.concatenate([features[obj]['vector_reid'], features[obj]['vector_clip']])
            for obj in unmatched_objects]

    clustering = OPTICS(min_samples=2, xi=0.05, min_cluster_size=2).fit(vecs)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1: continue
        if label not in clusters: clusters[label] = []
        clusters[label].append(unmatched_objects[idx])

    global_objects = {}
    for cl_id, objs in clusters.items():
        global_id = f'cluster_{cl_id}'

        appearances = []
        detections = []

        for obj in objs:
            cam_id, obj_id = obj
            appearances.append({
                'vector_reid': features[obj]['vector_reid'],
                'vector_clip': features[obj]['vector_clip'],
                'cam_id': cam_id,
                'obj_id': obj_id
            })

            for det in metadata[obj]:
                detections.append({
                    'cam_id': cam_id,
                    'frame_id': det['frame_id'],
                    'bbox': det['bbox'],
                    'timestamp': det['frame_id'] / config.fps
                })

        global_objects[global_id] = {
            'appearances': appearances,
            'avg_vector_reid': np.mean([a['vector_reid'] for a in appearances], axis=0),
            'avg_vector_clip': np.mean([a['vector_clip'] for a in appearances], axis=0),
            'metadata': {'detections': detections}
        }

    return global_objects