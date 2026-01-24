import json
import os

def find_single_image_ids(crops_base_path="data/crops", output_json="single_image_ids.json"):
    """
    Tìm các ID chỉ có 1 ảnh duy nhất trong crops và lưu vào file JSON.

    Args:
        crops_base_path: Đường dẫn đến thư mục crops
        output_json: Đường dẫn file JSON output

    Returns:
        Dict với cấu trúc: {seq_id: {cam_id: [list_of_single_ids]}}
    """
    from collections import defaultdict
    counts = 0

    print(f"Scanning crops directory: {crops_base_path}")

    result = defaultdict(lambda: defaultdict(list))

    # Loop qua tất cả các seq
    for seq_dir in os.listdir(crops_base_path):
        seq_path = os.path.join(crops_base_path, seq_dir)

        if not os.path.isdir(seq_path) or not seq_dir.startswith("seq_"):
            continue

        seq_id = seq_dir  # "seq_000", "seq_001", ...

        # Loop qua tất cả các camera
        for cam_dir in os.listdir(seq_path):
            cam_path = os.path.join(seq_path, cam_dir)

            if not os.path.isdir(cam_path) or not cam_dir.startswith("camera_"):
                continue

            cam_id = cam_dir  # "camera_1", "camera_2", ...

            # Đếm số ảnh của mỗi ID
            id_counts = defaultdict(int)

            for filename in os.listdir(cam_path):
                if not filename.endswith(".webp"):
                    continue

                # Parse tên file: "100001_000006.webp" -> id=100001, frame=000006
                try:
                    obj_id = filename.split("_")[0]
                    id_counts[obj_id] += 1
                except:
                    print(f"Cannot parse filename: {filename}")
                    continue

            # Lọc các ID chỉ có 1 ảnh
            single_ids = [obj_id for obj_id, count in id_counts.items() if count == 1]

            if single_ids:
                result[seq_id][cam_id] = sorted(single_ids)
                print(f"{seq_id}/{cam_id}: Found {len(single_ids)} IDs with single image")
                counts += len(single_ids)

    # Convert defaultdict sang dict thường để JSON serialize
    result_dict = {k: dict(v) for k, v in result.items()}

    # Lưu vào file JSON
    with open(output_json, "w") as f:
        json.dump(result_dict, f, indent=2)

    print(f"Saved single image IDs to {output_json}")

    return result_dict, counts

def move_single_images(json_file="single_image_ids.json",
                      crops_base="K:/Python/PersonRetrieval/data/crops",
                      output_base="K:/Python/PersonRetrieval/data/crops_single"):
    """
    Di chuyển các ảnh có single ID sang thư mục crops_single và xóa khỏi crops.

    Args:
        json_file: File JSON chứa danh sách single IDs
        crops_base: Thư mục crops gốc
        output_base: Thư mục đích crops_single
    """
    import shutil

    # Đọc file JSON
    with open(json_file, "r") as f:
        single_ids = json.load(f)

    total_moved = 0

    # Loop qua từng seq
    for seq_id, cameras in single_ids.items():
        # Loop qua từng camera
        for cam_id, ids in cameras.items():
            # Tạo thư mục đích
            src_dir = os.path.join(crops_base, seq_id, cam_id)
            dest_dir = os.path.join(output_base, seq_id, cam_id)
            os.makedirs(dest_dir, exist_ok=True)

            # Loop qua từng ID
            for obj_id in ids:
                # Tìm tất cả file ảnh của ID này
                for filename in os.listdir(src_dir):
                    if filename.startswith(f"{obj_id}_") and filename.endswith(".webp"):
                        src_file = os.path.join(src_dir, filename)
                        dest_file = os.path.join(dest_dir, filename)

                        # Di chuyển file (xóa khỏi crops gốc)
                        shutil.move(src_file, dest_file)
                        total_moved += 1
                        print(f"Moved: {seq_id}/{cam_id}/{filename}")

    print(f"Total files moved: {total_moved}")
    return total_moved

if __name__ == '__main__':
    # Tìm các ID chỉ có 1 ảnh
    # single_image_ids, counts = find_single_image_ids(
    #     crops_base_path="K:/Python/PersonRetrieval/data/crops",
    #     output_json="single_image_ids.json"
    # )
    # print(f"Found {counts} IDs with single image")

    moved = move_single_images(
        json_file="single_image_ids.json",
        crops_base="K:/Python/PersonRetrieval/data/crops",
        output_base="K:/Python/PersonRetrieval/data/crops_single"
    )