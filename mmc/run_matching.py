from data_loader import load_all_data
from matching import build_graph, perform_matching, apply_refined_clustering
import networkx as nx
import json
from logger import logger
import os
from collections import defaultdict
from database.upsert_qdrant import upsert_to_qdrant

def run_matching(camera_files):
    try:
        logger.info(f"Loading data from {len(camera_files)} cameras ...")
        metadata, features = load_all_data(camera_files)
        logger.info(f"Loaded {len(features)} objects ...")

        logger.info(f"Create graph ...")
        objects = list(features.keys())
        G = build_graph(objects, features, metadata)
        logger.info(f"Graph has {G.number_of_nodes()} nodes ..., {G.number_of_edges()} edges ...")

        logger.info(f"Matching ...")
        global_objects = perform_matching(G, features, metadata)
        logger.info(f"Matched {len(global_objects)} objects ...")

        matched_nodes = set(node for comp in nx.connected_components(G) for node in comp)
        unmatched = set(objects) - matched_nodes
        unmatched_objects = list(unmatched)
        logger.info(f"{len(unmatched_objects)} objects not matching")

        logger.info("Clustering for unmatched and update global_objects ...")
        clustered_objects = apply_refined_clustering(unmatched_objects, features, metadata)
        logger.info(f"Cluster {len(clustered_objects)} objects ...")
        global_objects.update(clustered_objects)

        return global_objects
    except Exception as ex:
        logger.error(f"Matching failed: {ex}")
        raise

def load_camera_files(path="data/metadata/metadata_features_files.json"):
    if not os.path.exists(path):
        logger.error(f"File {path} not found.")

    with open(path, "r") as f:
        files = json.load(f)

    file_paths = [tuple(x) for x in files["cam_files"]]
    return file_paths

def export_mot(global_objects, output_path):
    """Tạo file txt MOT17 để đánh giá"""
    # Chuẩn bị dữ liệu
    buffer = defaultdict(list)
    for global_id ,global_object in global_objects.items():
        if "global" in global_id:
            gid = int(global_id.split("_")[-1])
        else:
            gid = int(global_id.split("_")[-1]) * 1000
            if gid == 0:
                gid = 1000
        for app in global_object['apperances']:
            seq_id = app['seq_id']
            cam_id = app['cam_id']

            for dectection in app['detections']:
                frame_id = dectection['frame_id']
                x, y, w, h = dectection['bbox']

                line = [frame_id, gid, x, y, w, h, -1, -1, -1, -1]
                buffer[(seq_id, cam_id)].append(line)

    # Ghi vào file
    for (seq_id, cam_id), lines in buffer.items():
        seq_dir = os.path.join(output_path, f"seq_{seq_id}")
        os.makedirs(seq_dir, exist_ok=True)

        file_path = os.path.join(seq_dir, f"cam_{cam_id}.txt")
        lines.sort(key=lambda x: x[0])

        with open(file_path, "w") as f:
            for line in lines:
                f.write(",".join(map(str, line)) + "\n")

if __name__ == '__main__':
    camera_files = load_camera_files()
    logger.info("Running matching...")
    global_objects = run_matching(camera_files)
    logger.info("Exporting MOT17 ...")
    export_mot(global_objects, "data/preds")
    logger.info("Uploading QDrant ...")
    upsert_to_qdrant(global_objects)
    logger.info("Done.")