import numpy as np
from itertools import combinations
import pickle
from data_loader import load_all_data
import json
import os

def compute_tracklet_intervals(all_metadatas):
    """
    Returns:
        intervals_dict[(cam_id, object_id)] = (t_start, t_end)
    """
    intervals = {}

    for key, records in all_metadatas.items():
        frame_ids = [d["frame_id"] for d in records]
        t_start = min(frame_ids)
        t_end = max(frame_ids)
        intervals[key] = (t_start, t_end)

    return intervals

def build_clustering_input_weighted(
        all_metadatas,
        all_features
):
    """
    Returns:
        reid_embeddings: np.ndarray (N, D1)
        clip_embeddings: np.ndarray (N, D2)
        intervals: List[(t_start, t_end)]
        tracklet_keys: List[(cam_id, object_id)]
    """
    intervals_dict = compute_tracklet_intervals(all_metadatas)

    reid_embeddings = []
    clip_embeddings = []
    intervals = []
    tracklet_keys = []

    for key, feat in all_features.items():
        if key not in intervals_dict:
            continue

        reid_embeddings.append(
            np.array(feat["vector_reid"], dtype=np.float32)
        )
        clip_embeddings.append(
            np.array(feat["vector_clip"], dtype=np.float32)
        )

        intervals.append(intervals_dict[key])
        tracklet_keys.append(key)

    reid_embeddings = np.stack(reid_embeddings)
    clip_embeddings = np.stack(clip_embeddings)

    # L2 normalize
    reid_embeddings /= np.linalg.norm(reid_embeddings, axis=1, keepdims=True)
    clip_embeddings /= np.linalg.norm(clip_embeddings, axis=1, keepdims=True)

    return reid_embeddings, clip_embeddings, intervals, tracklet_keys

# --------------------------
def time_compatible(t1, t2):
    t1_start, t1_end = t1
    t2_start, t2_end = t2
    return (t2_start > t1_end) or (t1_start > t2_end)


def cluster_time_compatible(C1, C2, intervals):
    for i in C1:
        for j in C2:
            if not time_compatible(intervals[i], intervals[j]):
                return False
    return True


def weighted_similarity(i, j, reid_embs, clip_embs, w_reid=0.7):
    w_clip = 1.0 - w_reid

    sim_reid = np.dot(reid_embs[i], reid_embs[j])
    sim_clip = np.dot(clip_embs[i], clip_embs[j])

    return w_reid * sim_reid + w_clip * sim_clip


def cluster_similarity_weighted(
        C1,
        C2,
        reid_embs,
        clip_embs,
        w_reid=0.7
):
    max_sim = -1.0
    for i in C1:
        for j in C2:
            s = weighted_similarity(
                i, j,
                reid_embs,
                clip_embs,
                w_reid
            )
            if s > max_sim:
                max_sim = s
    return max_sim


def agglomerative_with_time_constraint_weighted(
        reid_embs,
        clip_embs,
        intervals,
        tau=0.75,
        w_reid=0.7
):
    clusters = [{i} for i in range(len(reid_embs))]

    while True:
        best_pair = None
        best_sim = tau

        for i, j in combinations(range(len(clusters)), 2):
            C1, C2 = clusters[i], clusters[j]

            if not cluster_time_compatible(C1, C2, intervals):
                continue

            sim = cluster_similarity_weighted(
                C1, C2,
                reid_embs,
                clip_embs,
                w_reid
            )

            if sim > best_sim:
                best_sim = sim
                best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        clusters[i] |= clusters[j]
        del clusters[j]

    return clusters


def match_tracklets_weighted(
        all_metadatas,
        all_features,
        tau=0.75,
        w_reid=0.7
):
    """
    Main function to perform tracklet matching with time constraints.

    Args:
        all_metadatas: dict mapping (cam_id, object_id) to list of metadata
        all_features: dict mapping (cam_id, object_id) to feature dict
        tau: similarity threshold for merging clusters
        w_reid: weight for ReID embeddings (0-1)

    Returns:
        List of clusters, each cluster is a set of indices
    """
    reid_embs, clip_embs, intervals, tracklet_keys = build_clustering_input_weighted(
        all_metadatas,
        all_features
    )

    clusters = agglomerative_with_time_constraint_weighted(
        reid_embs,
        clip_embs,
        intervals,
        tau=tau,
        w_reid=w_reid
    )

    # Map clusters back to tracklet keys
    result = []
    for cluster in clusters:
        tracklet_ids = [tracklet_keys[idx] for idx in cluster]
        result.append(tracklet_ids)

    return result


def save_results(clusters, output_path):
    """Save clustering results to file."""
    with open(output_path, 'wb') as f:
        pickle.dump(clusters, f)


def organize_images_by_cluster(clusters, crops_dir, output_dir):
    """
    Organize cropped images into cluster folders.

    Args:
        clusters: List of clusters from matching
        crops_dir: Path to crops directory (e.g., "data/crops/seq_000/camera_2")
        output_dir: Path to output directory (e.g., "data/cluster")
    """
    import shutil

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each cluster
    for cluster_idx, tracklet_ids in enumerate(clusters):
        # Create cluster folder
        cluster_folder = os.path.join(output_dir, f"person_{cluster_idx + 1:03d}")
        os.makedirs(cluster_folder, exist_ok=True)

        print(f"Processing cluster {cluster_idx + 1}/{len(clusters)}...")

        img_count = 0
        # Copy images from each tracklet in cluster
        for seq_id, cam_id, obj_id in tracklet_ids:
            # Search for images matching object_id pattern in crops_dir
            for img_file in os.listdir(crops_dir):
                if img_file.endswith('.webp') and str(obj_id) in img_file:
                    src = os.path.join(crops_dir, img_file)
                    # Rename to avoid conflicts
                    new_name = f"cam{cam_id}_obj{obj_id}_{img_file}"
                    dst = os.path.join(cluster_folder, new_name)
                    shutil.copy2(src, dst)
                    img_count += 1

        print(f"  - Cluster person_{cluster_idx + 1:03d}: {len(tracklet_ids)} tracklets, {img_count} images")

    print(f"\nDone! Organized {len(clusters)} clusters in {output_dir}")


def run_matching(
        crops_dir='data/crops/seq_000/camera_2',
        output_dir='data/cluster',
        output_path='matching_results.pkl',
        tau=0.875,
        w_reid=0.7
):
    """
    Run the complete matching pipeline and organize images.

    Args:
        crops_dir: path to cropped images
        output_dir: path to save organized clusters
        output_path: path to save matching results
        tau: similarity threshold
        w_reid: weight for ReID features
    """
    # Load data
    all_metadatas, all_features = load_all_data([
        ("data/new_metadata_objects/seq_000/seq_000_camera_2.txt",
         "data/new_feature_objects/seq_000/seq_000_camera_2.pkl")
    ])

    # Perform matching
    clusters = match_tracklets_weighted(
        all_metadatas,
        all_features,
        tau=tau,
        w_reid=w_reid
    )

    # Save results
    save_results(clusters, output_path)

    print(f"Found {len(clusters)} unique objects")
    print(f"Results saved to {output_path}")

    # Organize images by cluster
    organize_images_by_cluster(clusters, crops_dir, output_dir)

    return clusters


if __name__ == "__main__":
    clusters = run_matching(
        crops_dir='data/crops/seq_000/camera_2',
        output_dir='data/new_cluster/seq_000_camera_2',
        output_path='single_cam_results_0_2.pkl',
        tau=0.875,
        w_reid=0.7
    )

