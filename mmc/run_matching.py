from .data_loader import load_all_data
from .matching import build_graph, perform_matching, apply_refined_clustering
import networkx as nx
import json
from logger import get_logger
import os
from collections import defaultdict
from database.upsert_qdrant import upsert_to_qdrant
import shutil

logger = get_logger(name=__name__)

def run_matching(camera_files):
    try:
        logger.info(f"Loading data from {len(camera_files)} camera files ...")
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
            gid = int(global_id.split("_")[-1]) + 1
        else:
            gid = int(global_id.split("_")[-1]) * 1000
            if gid == 0:
                gid = 1000
        for app in global_object['appearances']:
            seq_id = app['seq_id']
            cam_id = app['cam_id']

            for dectection in app['detections']:
                frame_id = dectection['frame_id']
                x, y, w, h = dectection['bbox']

                line = [frame_id, gid, x, y, w, h, 1, -1, -1, -1]
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

def save_global_id_to_txt(global_objects, save_path):
    """
    Save mapping: global_id -> list of (seq_id, cam_id, obj_id) to a txt file.
    """

    with open(save_path, "w") as f:
        for global_id, global_object in global_objects.items():
            apps = global_object.get("appearances", [])

            f.write(f"global_id {global_id}:\n")
            for a in apps:
                seq_id = a["seq_id"]
                cam_id = a["cam_id"]
                obj_id = a["obj_id"]
                f.write(f"  seq={seq_id} cam={cam_id} obj={obj_id}\n")

            f.write("\n")

def organize_crops_by_global_id(global_objects, crops_base_path="data/crops", output_base_path="data/id_objects_all"):
    """
    Tổ chức ảnh crops theo global_id.
    Mỗi global_id sẽ có 1 thư mục riêng chứa tất cả crops của các appearances.

    Args:
        global_objects: Dict kết quả từ run_matching
        crops_base_path: Đường dẫn thư mục chứa crops gốc (data/crops/seq_xxx/cam_x/)
        output_base_path: Đường dẫn thư mục output (data/id_objects)
    """
    os.makedirs(output_base_path, exist_ok=True)
    logger.info(f"Organizing crops by global_id into {output_base_path}...")

    for global_id, global_object in global_objects.items():
        # Tạo thư mục cho global_id
        global_id_dir = os.path.join(output_base_path, str(global_id))
        os.makedirs(global_id_dir, exist_ok=True)

        appearances = global_object.get('appearances', [])

        for app in appearances:
            seq_id = app['seq_id']
            cam_id = app['cam_id']
            obj_id = app['obj_id']

            # Đường dẫn đến thư mục crops của camera này
            cam_crops_dir = os.path.join(crops_base_path, f"seq_{seq_id:03d}", f"camera_{cam_id}")

            if not os.path.exists(cam_crops_dir):
                logger.warning(f"Crops directory not found: {cam_crops_dir}")
                continue

            # Tìm và copy tất cả ảnh crops của object này
            for filename in os.listdir(cam_crops_dir):
                # Kiểm tra nếu file thuộc obj_id này
                if f"{obj_id}_" in filename or f"{obj_id:04d}" in filename:
                    src_path = os.path.join(cam_crops_dir, filename)

                    # Tạo tên file mới để tránh trùng lặp: seq_cam_originalname
                    new_filename = f"seq{seq_id:03d}_cam{cam_id}_{filename}"
                    dst_path = os.path.join(global_id_dir, new_filename)

                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")

        num_crops = len(os.listdir(global_id_dir)) if os.path.exists(global_id_dir) else 0
        logger.info(f"Global_id {global_id}: {num_crops} crops organized")

    logger.info("Crops organization completed.")


if __name__ == '__main__':
    camera_files = load_camera_files()
    logger.info("Running matching...")
    global_objects = run_matching(camera_files)
    # Export global_id
    save_global_id_to_txt(global_objects, "data/global_id_v2.txt")
    logger.info("Exporting MOT17 ...")
    export_mot(global_objects, "data/pred")
    # Tổ chức crops theo global_id
    logger.info("Organizing crops by global_id...")
    organize_crops_by_global_id(global_objects, crops_base_path="data/crops", output_base_path="data/id_objects_all")
    logger.info("Uploading QDrant ...")
    upsert_to_qdrant(global_objects, drop=True)
    logger.info("Done.")