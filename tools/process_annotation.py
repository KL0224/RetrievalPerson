import os
from pathlib import Path
import json
from collections import defaultdict

# Đường dẫn tới thư mục annotations
BASE_DIR = Path(__file__).resolve().parent.parent
annotation_dir = os.path.join(BASE_DIR, 'data','annotations')
# Thư mục lưu thông tin
metadata_dir = os.path.join(BASE_DIR, 'data', 'metadata')
metadata_timestamp_path = os.path.join(metadata_dir, "timestamp.json")
filepath = os.path.join(metadata_dir, "filepaths.json")

def extract_timestam(anno_dir):
    metadata_timestamp = {}
    for seq in os.listdir(anno_dir):
        for filepath in os.listdir(os.path.join(anno_dir, seq)):
            metadata_timestamp[seq] = filepath.split('.')[0].split("camera_1_")[-1]
            break

    return metadata_timestamp

def extract_filepath(anno_dir):
    list_filepath = {}
    for seq in os.listdir(anno_dir):
        if seq not in list_filepath:
            list_filepath[seq] = []
        for filepath in os.listdir(os.path.join(anno_dir, seq)):
            list_filepath[seq].append(filepath)

    return list_filepath

if __name__ == '__main__':
    metadata_timestamp = extract_timestam(annotation_dir)
    with open(metadata_timestamp_path, 'w') as f:
        json.dump(metadata_timestamp, f, indent=4, ensure_ascii=False)

    filepaths = extract_filepath(annotation_dir)
    with open(filepath, 'w') as f:
        json.dump(filepaths, f, indent=4, ensure_ascii=False)



