import numpy as np
import os
import glob
import pickle
import shutil
from itertools import combinations
from pathlib import Path


class Node:
    def __init__(self):
        self.id = None
        self.seq_id = None
        self.cam_id = None
        self.track_keys = []  # List of (seq_id, cam_id, obj_id)
        self.reid_average = None
        self.clip_average = None

    def compute_averages(self, all_features):
        reid_vecs = []
        clip_vecs = []

        for key in self.track_keys:
            feat = all_features[key]
            reid_vecs.append(feat['reid'])
            clip_vecs.append(feat['clip'])

        self.reid_average = np.mean(reid_vecs, axis=0)
        self.reid_average /= np.linalg.norm(self.reid_average)

        self.clip_average = np.mean(clip_vecs, axis=0)
        self.clip_average /= np.linalg.norm(self.clip_average)


def weighted_similarity_nodes(node1, node2, w_reid=0.7):
    """Compute weighted similarity between two nodes."""
    w_clip = 1.0 - w_reid

    sim_reid = np.dot(node1.reid_average, node2.reid_average)
    sim_clip = np.dot(node1.clip_average, node2.clip_average)

    return w_reid * sim_reid + w_clip * sim_clip


def get_dynamic_threshold(node1, node2, tau_same_seq=0.875, tau_diff_seq=0.85):
    """Get threshold based on whether nodes are from same sequence."""
    if node1.seq_id == node2.seq_id:
        return tau_same_seq
    else:
        return tau_diff_seq


def cluster_similarity_max(cluster1, cluster2, nodes, w_reid=0.7):
    """Compute maximum similarity between two clusters."""
    max_sim = -1.0
    for i in cluster1:
        for j in cluster2:
            sim = weighted_similarity_nodes(nodes[i], nodes[j], w_reid)
            if sim > max_sim:
                max_sim = sim
    return max_sim


def agglomerative_clustering_global(nodes, tau_same_seq=0.9, tau_diff_seq=0.9, w_reid=0.7):
    """
    Agglomerative clustering without time constraints.
    Uses dynamic threshold based on sequence compatibility.
    Skips comparison between nodes from same sequence AND same camera.
    """
    # Initialize each node as a cluster
    clusters = [{i} for i in range(len(nodes))]

    print(f"Starting clustering with {len(clusters)} initial clusters...")

    iteration = 0
    while True:
        best_pair = None
        best_sim = -1.0
        best_threshold = None

        # Find best pair to merge
        for i, j in combinations(range(len(clusters)), 2):
            cluster1, cluster2 = clusters[i], clusters[j]

            # ✅ NEW: Skip if ALL nodes in both clusters are from same seq AND same camera
            seqs_cams1 = {(nodes[idx].seq_id, nodes[idx].cam_id) for idx in cluster1}
            seqs_cams2 = {(nodes[idx].seq_id, nodes[idx].cam_id) for idx in cluster2}

            # If there's any overlap in (seq_id, cam_id) pairs, skip this pair
            if seqs_cams1 & seqs_cams2:
                continue

            # Compute similarity
            sim = cluster_similarity_max(cluster1, cluster2, nodes, w_reid)

            # Determine threshold based on sequences in clusters
            threshold = tau_diff_seq  # Default to cross-sequence threshold

            # Check if all nodes in both clusters are from same sequence
            seqs1 = {nodes[idx].seq_id for idx in cluster1}
            seqs2 = {nodes[idx].seq_id for idx in cluster2}

            # If there's overlap in sequences, use same-seq threshold
            if seqs1 & seqs2:
                threshold = tau_same_seq

            # Update best pair if similarity exceeds threshold
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_pair = (i, j)
                best_threshold = threshold

        if best_pair is None:
            break

        # Merge best pair
        i, j = best_pair
        clusters[i] |= clusters[j]
        del clusters[j]

        iteration += 1
        if iteration % 10 == 0:
            print(
                f"  Iteration {iteration}: {len(clusters)} clusters remaining (merged with sim={best_sim:.4f}, threshold={best_threshold})")

    print(f"Clustering complete: {len(clusters)} global IDs found")
    return clusters



def save_global_results(clusters, nodes, output_path):
    """Save global clustering results."""
    results = []
    for cluster in clusters:
        # Collect all track_keys from nodes in cluster
        global_tracks = []
        for node_idx in cluster:
            node = nodes[node_idx]
            global_tracks.extend(node.track_keys)
        results.append(global_tracks)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {output_path}")
    return results


def organize_global_images(global_clusters, crops_base_dir='data/crops', output_dir='data/global_ids'):
    """
    Organize images by global ID across all sequences and cameras.

    Args:
        global_clusters: List of lists, each containing track_keys (seq_id, cam_id, obj_id)
        crops_base_dir: Base directory containing crops organized by seq/camera
        output_dir: Output directory for global_ids
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOrganizing images into global IDs...")
    print(f"Source: {crops_base_dir}")
    print(f"Destination: {output_dir}")

    total_images_copied = 0

    for global_id, track_keys in enumerate(global_clusters):
        # Create folder for this global ID
        global_folder = os.path.join(output_dir, f"global_id_{global_id:03d}")
        os.makedirs(global_folder, exist_ok=True)

        images_in_cluster = 0

        # Process each track in the cluster
        for seq_id, cam_id, obj_id in track_keys:
            # Construct path to crops for this sequence/camera
            seq_folder = f"seq_{seq_id:03d}"
            cam_folder = f"camera_{cam_id}"
            crops_dir = os.path.join(crops_base_dir, seq_folder, cam_folder)

            if not os.path.exists(crops_dir):
                print(f"  Warning: Directory not found: {crops_dir}")
                continue

            # Find all images for this object
            # Format: {cam_id*100000 + obj_id}_{frame_id}.webp
            obj_prefix = f"{obj_id}_"

            for img_file in os.listdir(crops_dir):
                if img_file.startswith(obj_prefix) and img_file.endswith('.webp'):
                    src_path = os.path.join(crops_dir, img_file)

                    # Create descriptive filename
                    frame_id = img_file.split('_')[1].replace('.webp', '')
                    new_filename = f"seq_{seq_id:03d}_cam_{cam_id}_obj_{obj_id:05d}_frame_{frame_id}.webp"
                    dst_path = os.path.join(global_folder, new_filename)

                    # Copy image
                    shutil.copy2(src_path, dst_path)
                    images_in_cluster += 1

        total_images_copied += images_in_cluster
        print(f"  Global ID {global_id:03d}: {len(track_keys)} tracks, {images_in_cluster} images")

    print(f"\nTotal: {len(global_clusters)} global IDs, {total_images_copied} images copied")


def load_nodes_from_clusters():
    """Load all nodes from single-camera clustering results."""
    clusters_dir = 'new_clusters'
    seqs = sorted(glob.glob(os.path.join(clusters_dir, 'seq_*/')))
    cameras = ['camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_5', 'camera_6', 'camera_7']

    nodes = []
    node_id = 0

    print("Loading nodes from single-camera clusters...")

    for seq in seqs:
        seq_name = os.path.basename(os.path.normpath(seq))
        seq_id = int(seq_name.split('_')[-1])

        for cam in cameras:
            cam_id = int(cam.split('_')[-1])

            cluster_file = f'{clusters_dir}/results/{seq_name}_{cam}_results.pkl'
            if not os.path.exists(cluster_file):
                continue

            feature_file_path = os.path.join('data', 'new_feature_objects', seq_name, f"{seq_name}_{cam}.pkl")
            if not os.path.exists(feature_file_path):
                continue

            # Load clusters and features
            with open(cluster_file, "rb") as f:
                clusters = pickle.load(f)

            with open(feature_file_path, "rb") as f:
                all_features = pickle.load(f)

            # Create node for each cluster
            for cluster in clusters:
                node = Node()
                node.id = node_id
                node.seq_id = seq_id
                node.cam_id = cam_id
                node.track_keys = cluster
                node.compute_averages(all_features)
                nodes.append(node)
                node_id += 1

        print(f"  {seq_name}: {node_id} nodes loaded")

    print(f"Total nodes loaded: {len(nodes)}")
    return nodes


def run_global_matching(
        tau_same_seq=0.875,
        tau_diff_seq=0.92,
        w_reid=0.7,
        crops_base_dir='data/crops',
        output_dir='data/global_ids',
        results_file='global_matching_results.pkl'
):
    """
    Main function to run global matching across all sequences and cameras.
    """
    print("=" * 80)
    print("GLOBAL MULTI-CAMERA MULTI-SEQUENCE MATCHING")
    print("=" * 80)

    # Step 1: Load all nodes
    nodes = load_nodes_from_clusters()

    if len(nodes) == 0:
        print("No nodes found! Please run single-camera matching first.")
        return

    print(f"\nConfiguration:")
    print(f"  - Same sequence threshold: {tau_same_seq}")
    print(f"  - Different sequence threshold: {tau_diff_seq}")
    print(f"  - ReID weight: {w_reid}")
    print(f"  - CLIP weight: {1.0 - w_reid}")

    # Step 2: Perform global clustering
    print("\n" + "=" * 80)
    print("STEP 1: Global Clustering")
    print("=" * 80)
    global_clusters = agglomerative_clustering_global(
        nodes,
        tau_same_seq=tau_same_seq,
        tau_diff_seq=tau_diff_seq,
        w_reid=w_reid
    )

    # Step 3: Convert clusters to track_keys format
    print("\n" + "=" * 80)
    print("STEP 2: Converting to Track Keys")
    print("=" * 80)
    global_track_clusters = []
    for cluster in global_clusters:
        tracks = []
        for node_idx in cluster:
            tracks.extend(nodes[node_idx].track_keys)
        global_track_clusters.append(tracks)

    # Step 4: Save results
    print("\n" + "=" * 80)
    print("STEP 3: Saving Results")
    print("=" * 80)
    save_global_results(global_clusters, nodes, results_file)

    # Step 5: Organize images
    print("\n" + "=" * 80)
    print("STEP 4: Organizing Images")
    print("=" * 80)
    organize_global_images(global_track_clusters, crops_base_dir, output_dir)

    print("\n" + "=" * 80)
    print("GLOBAL MATCHING COMPLETE!")
    print("=" * 80)
    print(f"Found {len(global_track_clusters)} global identities")

    # Print statistics
    print("\nStatistics:")
    cluster_sizes = [len(cluster) for cluster in global_track_clusters]
    print(f"  - Min tracks per global ID: {min(cluster_sizes)}")
    print(f"  - Max tracks per global ID: {max(cluster_sizes)}")
    print(f"  - Average tracks per global ID: {np.mean(cluster_sizes):.2f}")

    return global_track_clusters


if __name__ == "__main__":
    global_clusters = run_global_matching(
        tau_same_seq=0.875,
        tau_diff_seq=0.92,
        w_reid=0.7,
        crops_base_dir='data/crops',
        output_dir='data/global_ids',
        results_file='global_matching_results_1.pkl'
    )
