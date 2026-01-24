import os
import re
import glob
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List, Any

import pandas as pd


def load_global_mapping(pkl_path: str) -> Dict[Tuple[str, int, int], int]:
    """
    Expected mapping key: (seq_id, cam_id, obj_id) -> global_id
    Supports either direct-dict pickle or wrapper dict containing that mapping.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        # direct mapping
        if all(isinstance(k, tuple) and len(k) == 3 for k in obj.keys()):
            return obj  # type: ignore[return-value]

        # try common wrapper keys
        for key in ("global_mapping", "mapping", "matches", "result", "results"):
            if key in obj and isinstance(obj[key], dict):
                m = obj[key]
                if all(isinstance(k, tuple) and len(k) == 3 for k in m.keys()):
                    return m  # type: ignore[return-value]


def _parse_seq_cam_from_path(path: str) -> Tuple[str, int]:
    """
    From: ...\\seq_000\\seq_000_camera_1.txt -> (seq_000, 1)
    """
    fname = os.path.basename(path)
    m = re.match(r"(seq_\d+)_camera_(\d+)\.txt$", fname)
    if not m:
        raise ValueError(f"Unexpected metadata filename: {path}")
    return m.group(1), int(m.group(2))


def _read_metadata_file(path: str) -> List[Dict[str, Any]]:
    """
    Tries to read the metadata file as CSV/TSV/whitespace-delimited.
    Expects at least: frame_id and bbox (either bbox as 4 cols or as a string/list).
    """
    # Try common delimiters
    last_err = None
    for sep in (",", "\t", r"\s+"):
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if len(df.columns) > 1:
                return df.to_dict("records")
        except Exception as e:
            last_err = e
            continue


def _extract_bbox(det: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Supports:
      - det["bbox"] as [x1,y1,x2,y2] list/tuple
      - det["bbox"] as string like "[x1, y1, x2, y2]"
      - columns: x1,y1,x2,y2
      - columns: xmin,ymin,xmax,ymax
      - columns: left,top,right,bottom
    """
    if "bbox" in det and det["bbox"] is not None:
        b = det["bbox"]
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        if isinstance(b, str):
            # extract numbers from string
            nums = re.findall(r"-?\d+(?:\.\d+)?", b)
            if len(nums) >= 4:
                x1, y1, x2, y2 = map(float, nums[:4])
                return x1, y1, x2, y2

    for keys in (("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax"), ("left", "top", "right", "bottom")):
        if all(k in det for k in keys):
            x1, y1, x2, y2 = (float(det[keys[0]]), float(det[keys[1]]), float(det[keys[2]]), float(det[keys[3]]))
            return x1, y1, x2, y2


def _extract_frame(det: Dict[str, Any]) -> int:
    for k in ("frame_id", "frame", "fid"):
        if k in det and det[k] is not None:
            return int(det[k])


def _extract_obj_id(det: Dict[str, Any]) -> int:
    for k in ("obj_id", "id", "track_id"):
        if k in det and det[k] is not None:
            return int(det[k])
    raise ValueError("Cannot find obj_id/id in detection record.")


def build_mot_predictions_from_metadata(
    metadata_root: str,
    global_mapping: Dict[Tuple[str, int, int], int],
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Returns: {(seq_id, cam_id): MOT-format DataFrame}
    """
    rows_by_seq_cam: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)

    pattern = os.path.join(metadata_root, "seq_*", "seq_*_camera_*.txt")
    for path in glob.glob(pattern):
        seq_id, cam_id = _parse_seq_cam_from_path(path)
        dets = _read_metadata_file(path)

        for det in dets:
            frame_id = _extract_frame(det)
            obj_id = _extract_obj_id(det)
            bbox = _extract_bbox(det)
            x1, y1, x2, y2 = bbox

            global_id = global_mapping.get((seq_id, cam_id, obj_id))
            if global_id is None:
                continue

            bb_left = x1
            bb_top = y1
            bb_width = x2 - x1
            bb_height = y2 - y1

            conf = float(det.get("conf", 1.0))

            rows_by_seq_cam[(seq_id, cam_id)].append(
                {
                    "frame": int(frame_id),
                    "id": int(global_id),
                    "bb_left": float(bb_left),
                    "bb_top": float(bb_top),
                    "bb_width": float(bb_width),
                    "bb_height": float(bb_height),
                    "conf": float(conf),
                    "x": -1,
                    "y": -1,
                    "z": -1,
                }
            )

    out: Dict[Tuple[str, int], pd.DataFrame] = {}
    for key, rows in rows_by_seq_cam.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df = df.sort_values(["frame", "id"], ascending=True)
        out[key] = df

    return out


def write_mot_files(predictions: Dict[Tuple[str, int], pd.DataFrame], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for (seq_id, cam_id), df in predictions.items():
        seq_dir = os.path.join(out_dir, seq_id)
        os.makedirs(seq_dir, exist_ok=True)

        out_path = os.path.join(seq_dir, f"{seq_id}_cam{cam_id}.txt")

        # MOTChallenge format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
        df.to_csv(
            out_path,
            index=False,
            header=False,
            columns=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"],
        )


def main() -> None:
    pkl_path = "global_matching_results_2 (1).pkl"
    metadata_root = os.path.join("data", "new_metadata_objects")
    out_dir = "mot_predictions"

    global_mapping = load_global_mapping(pkl_path)
    predictions = build_mot_predictions_from_metadata(metadata_root, global_mapping)
    write_mot_files(predictions, out_dir)

    print(f"Wrote {len(predictions)} MOT files into: {out_dir}")


if __name__ == "__main__":
    main()
