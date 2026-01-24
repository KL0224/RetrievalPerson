import json
import os
from collections import defaultdict
import sys
sys.path.append("..")

def extract_temporal_constraints(seq_root_path):
    # Dictionary lưu trữ: id -> {cam_id: (min_frame, max_frame)}
    person_history = defaultdict(dict)

    # 1. Đọc tất cả các file cam trong thư mục seq
    for file_name in sorted(os.listdir(seq_root_path)):
        if not file_name.endswith('.json'):
            continue

        # Giả định tên file là cam1.json, cam2.json... lấy số 1, 2 làm cam_id
        try:
            cam_id = file_name.split('_')[1]
        except:
            continue

        with open(os.path.join(seq_root_path, file_name), 'r') as f:
            data = json.load(f)

            # Duyệt qua từng frame (key "1", "2" trong JSON)
            for frame_str, objects in data.items():
                frame_idx = int(frame_str)
                for obj in objects:
                    obj_id = obj['id']

                    if obj_id not in person_history:
                        person_history[obj_id] = {}

                    if cam_id not in person_history[obj_id]:
                        # Khởi tạo (first_frame, last_frame)
                        person_history[obj_id][cam_id] = [frame_idx, frame_idx]
                    else:
                        # Cập nhật frame cuối cùng nhìn thấy
                        person_history[obj_id][cam_id][1] = max(person_history[obj_id][cam_id][1], frame_idx)

    # 2. Tính toán delta frame giữa các cặp camera
    # transitions[(cam_a, cam_b)] = [list các delta_f]
    transitions = defaultdict(list)

    for obj_id, cams in person_history.items():
        cam_ids = sorted(cams.keys())
        # So sánh các cặp camera mà ID này xuất hiện
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                cam_a = cam_ids[i]
                cam_b = cam_ids[j]

                # Tính Delta F: Frame bắt đầu cam sau - Frame kết thúc cam trước
                time_a = cams[cam_a]
                time_b = cams[cam_b]

                if time_b[0] > time_a[1]:  # Cam B xuất hiện sau Cam A
                    delta_f = time_b[0] - time_a[1]
                    transitions[(cam_a, cam_b)].append(delta_f)
                elif time_a[0] > time_b[1]:  # Cam A xuất hiện sau Cam B
                    delta_f = time_a[0] - time_b[1]
                    transitions[(cam_b, cam_a)].append(delta_f)

    # 3. Tổng hợp kết quả min_f, max_f
    constraints = {}
    for cam_pair, delta_list in transitions.items():
        constraints[cam_pair] = {
            "min_f": min(delta_list),
            "max_f": max(delta_list)
        }

    return constraints

def format_global_constraints(data):
    global_map = {}

    for seq, transitions in data.items():
        for (cam_a, cam_b), frames in transitions.items():
            pair = f"{cam_a}-{cam_b}"  # Format key thành string để lưu JSON
            if pair not in global_map:
                global_map[pair] = {"min": frames['min_f'], "max": frames['max_f']}
            else:
                global_map[pair]["min"] = min(global_map[pair]["min"], frames['min_f'])
                global_map[pair]["max"] = max(global_map[pair]["max"], frames['max_f'])

    # Thêm biên độ an toàn (Margin) 20% để tránh việc lọc quá chặt
    for pair in global_map:
        global_map[pair]["min"] = int(global_map[pair]["min"] * 0.8)
        global_map[pair]["max"] = int(global_map[pair]["max"] * 1.2)

    return global_map

all_sequences_constraints = {}
root_label_dir = "data/annotations"  # Thay bằng đường dẫn folder chứa seq_1, seq_2...

for seq_dir in os.listdir(root_label_dir):
    seq_path = os.path.join(root_label_dir, seq_dir)
    if os.path.isdir(seq_path):
        print(f"Processing {seq_dir}...")
        res = extract_temporal_constraints(seq_path)
        all_sequences_constraints[seq_dir] = res


# Lưu lại để dùng cho inference
formatted_constraints = format_global_constraints(all_sequences_constraints)
with open('../data/metadata/temporal_configs.json', 'w') as f:
    json.dump(formatted_constraints, f, indent=4)