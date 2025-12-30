import sys
from data_loader import load_all_data
from matching_utils import build_graph, perform_matching, apply_open_set_clustering
import networkx as nx
import json
from logger import logger

def run_matching(camera_files):
    try:
        logger.info(f"Loading data từ {len(camera_files)} cameras ...")
        metadata, features = load_all_data(camera_files)
        logger.info(f"Đã load {len(features)} đối tượng ...")

        logger.info(f"Tạo graph ...")
        objects = list(features.keys())
        G = build_graph(objects, features, metadata)
        logger.info(f"Graph có {G.number_of_nodes()} nodes ..., {G.number_of_edges()} edges ...")

        logger.info(f"Tiến hành matching ...")
        global_objects = perform_matching(G, features, metadata)
        logger.info(f"Đã matching được {len(global_objects)} đối tượng")

        matched_nodes = set(node for comp in nx.connected_components(G) for node in comp)
        unmatched = set(objects) - matched_nodes
        unmatched_objects = list(unmatched)
        logger.info(f"Còn {len(unmatched_objects)} đối tượng chưa matching")

        logger.info("Tiến hành clustering cho unmatched và update global_objects ...")
        clustered_objects = apply_open_set_clustering(unmatched_objects, features, metadata)
        global_objects.update(clustered_objects)

        return global_objects
    except Exception as ex:
        logger.error(f"Matching failed: {ex}")
        raise

def save_to_json(global_objects, file_path='global_objects.json'):
    objects = {}
    for global_id, data in global_objects.items():
        # Lưu vector trung bình để tham khảo nhanh
        objects[global_id] = {
            'avg_vector_reid': data['avg_vector_reid'].tolist(),
            'avg_vector_clip': data['avg_vector_clip'].tolist(),
            'appearances_count': len(data['appearances']),
            'metadata': data['metadata']
        }
    with open(file_path, 'w') as f:
        json.dump(objects, f, indent=4)
    print(f"Đã lưu global_objects vào {file_path}")

if __name__ == '__main__':
    global_objects = run_matching(sys.argv[1:])
    save_to_json(global_objects)