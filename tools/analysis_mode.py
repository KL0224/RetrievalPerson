import pickle
import glob
import os


def count_total_nodes(results_dir='new_clusters/results'):
    """
    Đọc tất cả file pkl trong thư mục results và đếm tổng số nodes (clusters).

    Returns:
        dict: Thống kê chi tiết theo sequence và tổng số nodes
    """
    # Tìm tất cả file pkl
    pkl_files = glob.glob(os.path.join(results_dir, '*.pkl'))

    total_nodes = 0
    stats = {}  # {seq_id: {cam_id: num_nodes}}

    print("=" * 80)
    print("COUNTING NODES FROM SINGLE-CAMERA CLUSTERING RESULTS")
    print("=" * 80)

    for pkl_file in sorted(pkl_files):
        # Parse filename: seq_000_camera_1_results.pkl
        filename = os.path.basename(pkl_file)
        parts = filename.replace('_results.pkl', '').split('_')
        seq_id = int(parts[1])  # seq_000 -> 0
        cam_id = int(parts[3])  # camera_1 -> 1

        # Load clusters từ file
        with open(pkl_file, 'rb') as f:
            clusters = pickle.load(f)

        # Mỗi cluster = 1 node
        num_nodes = len(clusters)
        total_nodes += num_nodes

        # Lưu statistics
        if seq_id not in stats:
            stats[seq_id] = {}
        stats[seq_id][cam_id] = num_nodes

        print(f"  {filename:<40} : {num_nodes:4d} nodes")

    print("=" * 80)
    print(f"TOTAL NODES: {total_nodes}")
    print("=" * 80)

    # In thống kê theo sequence
    print("\nStatistics by Sequence:")
    for seq_id in sorted(stats.keys()):
        seq_total = sum(stats[seq_id].values())
        cam_count = len(stats[seq_id])
        print(f"  Seq {seq_id:03d}: {seq_total:4d} nodes ({cam_count} cameras)")

    return total_nodes, stats


def inspect_node_structure(sample_file='new_clusters/results/seq_000_camera_1_results.pkl'):
    """
    Kiểm tra cấu trúc chi tiết của 1 file để hiểu format dữ liệu.
    """
    print("\n" + "=" * 80)
    print(f"INSPECTING SAMPLE FILE: {sample_file}")
    print("=" * 80)

    with open(sample_file, 'rb') as f:
        clusters = pickle.load(f)

    print(f"\nTotal clusters (nodes): {len(clusters)}")
    print(f"Data type: {type(clusters)}")

    # Kiểm tra vài clusters đầu tiên
    print("\nFirst 5 clusters:")
    for i, cluster in enumerate(clusters[:5]):
        print(f"\n  Cluster {i}:")
        print(f"    Type: {type(cluster)}")
        print(f"    Size: {len(cluster)} tracks")
        print(f"    Content: {cluster[:3]}...")  # Show first 3 tracks

        # Giải thích format của track_key
        if cluster:
            first_track = cluster[0]
            print(f"    Track format: (seq_id={first_track[0]}, cam_id={first_track[1]}, obj_id={first_track[2]})")


if __name__ == "__main__":
    # 1. Đếm tổng số nodes
    total_nodes, stats = count_total_nodes()

    # 2. Kiểm tra cấu trúc chi tiết
    inspect_node_structure()

    # 3. In tóm tắt
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total sequences: {len(stats)}")
    print(f"Total cameras: {sum(len(cams) for cams in stats.values())}")
    print(f"Total nodes (clusters): {total_nodes}")
    print(f"Average nodes per camera: {total_nodes / sum(len(cams) for cams in stats.values()):.2f}")
