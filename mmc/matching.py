import numpy as np
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from config import config

def compute_weighted_similarity(feat1, feat2, w_reid=0.7, w_clip=0.3):
    """
    Tính toán độ tương đồng có trọng số giữa 2 đối tượng.
    """
    # Tính Cosine Similarity cho ReID
    sim_reid = np.dot(feat1['vector_reid'], feat2['vector_reid']) / (
            np.linalg.norm(feat1['vector_reid']) * np.linalg.norm(feat2['vector_reid']) + 1e-6
    )

    # Tính Cosine Similarity cho CLIP
    sim_clip = np.dot(feat1['vector_clip'], feat2['vector_clip']) / (
            np.linalg.norm(feat1['vector_clip']) * np.linalg.norm(feat2['vector_clip']) + 1e-6
    )

    # Tổng hợp trọng số (ReID thường đáng tin cậy hơn về định danh cụ thể)
    return (w_reid * sim_reid) + (w_clip * sim_clip)


def is_temporal_valid(meta1, meta2, cam1, cam2, seq1, seq2):
    """
    Kiểm tra ràng buộc thời gian dựa trên file config.
    Xử lý cả trường hợp Overlap và Rời rạc.
    """
    # Nếu không cùng seq thì sẽ coi như là True để bỏ qua
    if seq1 != seq2:
        return True

    # Lấy frame đầu và cuối của mỗi tracklet
    t1_start = min(d['frame_id'] for d in meta1)
    t1_end = max(d['frame_id'] for d in meta1)
    t2_start = min(d['frame_id'] for d in meta2)
    t2_end = max(d['frame_id'] for d in meta2)

    pair_key = f"{cam1}-{cam2}"
    inv_pair_key = f"{cam2}-{cam1}"

    # TRƯỜNG HỢP 1: OVERLAP (Vùng nhìn chồng lấn)
    if not (t2_start > t1_end or t1_start > t2_end):
        overlap_duration = min(t1_end, t2_end) - max(t1_start, t2_start)

        # Chỉ cho phép overlap nếu là spatial neighbors
        if cam2 not in config.spatial_neighbors.get(cam1, []):
            return False

        # Overlap 3s
        return overlap_duration <= 100

    # TRƯỜNG HỢP 2: DISJOINT (Rời rạc - Đi từ cam này sang cam kia)
    if t2_start > t1_end:
        delta = t2_start - t1_end
        key = pair_key
    else:
        delta = t1_start - t2_end
        key = inv_pair_key

    stats = config.temporal_neighbors.get(key)

    if stats:
        return stats['min'] <= delta <= stats['max']

    # Không có thống kê → chỉ cho phép nếu spatial neighbor & delta nhỏ
    if cam2 in config.spatial_neighbors.get(cam1, []):
        return delta <= 2000  # ~1 phút

    return False


def build_graph(objects, features, metadata):
    """
    Xây dựng đồ thị Matching.
    Nodes: Các tracklet (seq_id, cam_id, obj_id).
    Edges: Được tạo nếu thỏa mãn Spatial -> Temporal -> Visual Threshold.
    """
    G = nx.Graph()
    obj_keys = list(objects)

    for i in range(len(obj_keys)):
        seq1, cam1, obj1 = obj_keys[i]

        for j in range(i + 1, len(obj_keys)):
            seq2, cam2, obj2 = obj_keys[j]

            # Bỏ qua nếu cùng một camera
            if seq1 == seq2 and cam1 == cam2: continue

            # SPATIAL FILTER (Lọc không gian)
            neighbors = config.spatial_neighbors.get(f"C{cam1}", [])
            if f"C{cam2}" not in neighbors:
                continue

            # TEMPORAL FILTER (Lọc thời gian)
            if not is_temporal_valid(metadata[(seq1, cam1, obj1)], metadata[(seq1, cam2, obj2)], cam1, cam2, seq1, seq2):
                continue

            # SIMILARITY
            sim = compute_weighted_similarity(features[(seq1, cam1, obj1)], features[(seq2, cam2, obj2)])

            if sim > config.similarity_threshold:
                G.add_edge((cam1, obj1), (cam2, obj2), weight=sim)

    return G

def perform_matching(G, features, metadata):
    """
    Gom nhóm các thành phần liên thông (Connected Components) thành Global ID.
    """
    components = list(nx.connected_components(G))
    global_objects = {}

    for idx, component in enumerate(components):
        global_id = f"global_{idx}"
        global_objects[global_id] = _aggregate_cluster(component, features, metadata)

    return global_objects

def apply_refined_clustering(unmatched_objects, features, metadata):
    """
    Sử dụng Agglomerative Clustering để xử lý các object chưa match.
    """
    if not unmatched_objects:
        return {}

    # Chỉ dùng Vector ReID để gom nhóm (chính xác hơn khi chạy Clustering đơn thuần)
    vecs = [features[obj]['vector_reid'] for obj in unmatched_objects]

    # Distance Threshold = 1 - Similarity Threshold
    # linkage='average' giúp gom nhóm ổn định hơn single/complete
    thresh = 1.0 - config.similarity_threshold
    # Đảm bảo thresh dương
    thresh = max(0.01, thresh)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=thresh,
        metric='cosine',
        linkage='average'
    ).fit(vecs)

    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters: clusters[label] = []
        clusters[label].append(unmatched_objects[idx])

    global_objects = {}
    for cl_id, objs in clusters.items():
        # Chỉ tạo nhóm nếu có ít nhất 1 phần tử (thực tế Agglomerative luôn nhóm >0)
        global_id = f'cluster_{cl_id}'
        global_objects[global_id] = _aggregate_cluster(objs, features, metadata)

    return global_objects


def _aggregate_cluster(component, features, metadata):
    """Hàm phụ trợ để gom dữ liệu của một nhóm (cluster/component) lại thành format đầu ra."""
    appearances = []
    for node in component:
        seq_id, cam_id, obj_id = node

        node_detections = []
        # Lấy thống tin frame/box của object
        for det in metadata[node]:
            node_detections.append({
                'frame_id': det['frame_id'],
                'bbox': det['bbox']
            })

        # Lưu thông tin feature gốc
        appearances.append({
            'vector_reid': features[node]['vector_reid'],
            'vector_clip': features[node]['vector_clip'],
            'seq_id': seq_id,
            'cam_id': cam_id,
            'obj_id': obj_id,
            'detections': node_detections
        })


    return {
        'appearances': appearances,
        'avg_vector_reid': np.mean([a['vector_reid'] for a in appearances], axis=0), # Nếu cần
        'avg_vector_clip': np.mean([a['vector_clip'] for a in appearances], axis=0) # Nếu cần
    }