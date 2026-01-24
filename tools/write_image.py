import pickle
import shutil
import os

def recreate_global_images_from_pkl(
        pkl_path='global_matching_results.pkl',
        crops_base_dir='data/crops',
        output_dir='data/global_ids_1'
):
    """
    Tạo lại thư mục ảnh từ file pickle đã lưu.
    """
    print("=" * 80)
    print("RECREATING GLOBAL IMAGES FROM PICKLE FILE")
    print("=" * 80)

    # Load global clusters
    print(f"\nLoading clusters from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        global_clusters = pickle.load(f)

    print(f"Loaded {len(global_clusters)} global clusters")

    # Organize images
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOrganizing images...")
    print(f"Source: {crops_base_dir}")
    print(f"Destination: {output_dir}")

    total_images_copied = 0

    for global_id, track_keys in enumerate(global_clusters):
        global_folder = os.path.join(output_dir, f"global_id_{global_id:03d}")
        os.makedirs(global_folder, exist_ok=True)

        images_in_cluster = 0

        for seq_id, cam_id, obj_id in track_keys:
            # Construct path
            crops_dir = os.path.join(crops_base_dir, f"seq_{seq_id:03d}", f"camera_{cam_id}")
            if not os.path.exists(crops_dir):
                continue

            # Find images: {obj_id}_*.webp
            obj_prefix = f"{obj_id}_"

            for img_file in os.listdir(crops_dir):
                if img_file.startswith(obj_prefix) and img_file.endswith('.webp'):
                    src_path = os.path.join(crops_dir, img_file)

                    # Parse frame_id
                    frame_id = img_file.split('_')[1].replace('.webp', '')

                    new_filename = f"seq_{seq_id:03d}_cam_{cam_id}_obj_{obj_id:05d}_frame_{frame_id}.webp"
                    dst_path = os.path.join(global_folder, new_filename)

                    shutil.copy2(src_path, dst_path)
                    images_in_cluster += 1

        total_images_copied += images_in_cluster

        if images_in_cluster > 0:
            print(f"  Global ID {global_id:03d}: {len(track_keys)} tracks, {images_in_cluster} images")

    print("\n" + "=" * 80)
    print(f"COMPLETE!")
    print(f"Total global IDs: {len(global_clusters)}")
    print(f"Total images copied: {total_images_copied}")
    print("=" * 80)

recreate_global_images_from_pkl()

