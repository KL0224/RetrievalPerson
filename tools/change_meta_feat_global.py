import os
import json
import pickle
from pathlib import Path
from typing import Dict, Set, Any, Iterable, Tuple


def _parse_camera_key(camera_key: str) -> str:
    """
    Convert `camera_1` -> `1` for filename matching `seq_000_camera_1.pkl` / `.txt`.
    """
    if not camera_key.startswith("camera_"):
        raise ValueError(f"Unexpected camera key: {camera_key}")
    return camera_key.split("_", 1)[1]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_removal_map(ids_json: Dict[str, Any]) -> Dict[Tuple[str, str], Set[str]]:
    """
    Map (seq_id, cam_id_str) -> set(object_id_str) to remove.
    cam_id_str is like `1`, `2`, ...
    """
    removal: Dict[Tuple[str, str], Set[str]] = {}
    for seq_id, cams in ids_json.items():
        if not isinstance(cams, dict):
            continue
        for camera_key, obj_ids in cams.items():
            cam_id = _parse_camera_key(camera_key)
            if not isinstance(obj_ids, list):
                continue
            removal[(seq_id, cam_id)] = set(str(x) for x in obj_ids)
    return removal


def _filter_metadata_txt(in_path: Path, out_path: Path, remove_ids: Set[str]) -> int:
    """
    Metadata line format: seq_id, cam_id, frame_id, object_id, ...
    We remove the line if 4th field (object_id) is in remove_ids.
    Returns number of kept lines.
    """
    _ensure_parent_dir(out_path)
    kept = 0

    with in_path.open("r", encoding="utf-8", errors="replace") as f_in, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:
        for line in f_in:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue

            parts = [p.strip() for p in raw.split(" ")]
            if len(parts) < 4:
                # Keep malformed/short lines unchanged
                f_out.write(raw + "\n")
                kept += 1
                continue

            object_id = parts[3]
            if object_id in remove_ids:
                continue

            f_out.write(raw + "\n")
            kept += 1

    return kept


def _iter_feature_records(obj: Any) -> Iterable[Tuple[Any, Any]]:
    """
    Try to iterate common pickle layouts:
    \- dict: values are feature vectors, keys include object_id or tuple keys.
    \- list/tuple: elements could be (key, feat) or dict-like items
    \- any iterable of pairs
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
        return

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, tuple) and len(item) == 2:
                yield item[0], item[1]
            elif isinstance(item, dict):
                for k, v in item.items():
                    yield k, v
            else:
                # Unknown element shape; skip
                continue
        return

    # Unknown container
    return


def _key_object_id(key: Any) -> str | None:
    """
    Extract object\_id from a feature record key.
    Expected: (seq_id, cam_id, object_id) or similar.
    """
    if isinstance(key, tuple) and len(key) >= 3:
        return str(key[2])
    return None


def _filter_feature_pkl(in_path: Path, out_path: Path, remove_ids: Set[str]) -> None:
    """
    Feature pickle expected: per record key is (seq_id, cam_id, object_id) and value is feature vector.
    Keeps the same container type when possible:
    \- dict -> dict filtered by key[2]
    \- list/tuple -> list filtered by element[0][2] if element is (key, value)
    Otherwise: rebuild as list of (key, value).
    """
    _ensure_parent_dir(out_path)

    with in_path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            obj_id = _key_object_id(k)
            if obj_id is not None and obj_id in remove_ids:
                continue
            new_data[k] = v

        with out_path.open("wb") as f:
            pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    if isinstance(data, (list, tuple)):
        new_list = []
        for item in data:
            if isinstance(item, tuple) and len(item) == 2:
                k, v = item
                obj_id = _key_object_id(k)
                if obj_id is not None and obj_id in remove_ids:
                    continue
                new_list.append(item)
            else:
                # If unknown, keep item as-is
                new_list.append(item)

        out_obj = type(data)(new_list) if isinstance(data, tuple) else new_list
        with out_path.open("wb") as f:
            pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    # Fallback: try to iterate pairs and rebuild
    new_pairs = []
    for k, v in _iter_feature_records(data):
        obj_id = _key_object_id(k)
        if obj_id is not None and obj_id in remove_ids:
            continue
        new_pairs.append((k, v))

    with out_path.open("wb") as f:
        pickle.dump(new_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    # Input roots
    ids_path = Path(r"K:\Python\PersonRetrieval\single_image_ids.json")
    feature_root = Path(r"K:\Python\PersonRetrieval\data\feature_objects")
    metadata_root = Path(r"K:\Python\PersonRetrieval\data\metadata_objects")

    # Output roots
    new_feature_root = Path(r"K:\Python\PersonRetrieval\data\new_feature_objects")
    new_metadata_root = Path(r"K:\Python\PersonRetrieval\data\new_metadata_objects")

    with ids_path.open("r", encoding="utf-8") as f:
        ids_json = json.load(f)

    removal_map = _build_removal_map(ids_json)

    # Process only seq/cam that exist in ids.json
    for (seq_id, cam_id), remove_ids in removal_map.items():
        # Paths preserve existing structure: `seq_000\seq_000_camera_1.pkl` and `.txt`
        in_feature = feature_root / seq_id / f"{seq_id}_camera_{cam_id}.pkl"
        in_metadata = metadata_root / seq_id / f"{seq_id}_camera_{cam_id}.txt"

        out_feature = new_feature_root / seq_id / f"{seq_id}_camera_{cam_id}.pkl"
        out_metadata = new_metadata_root / seq_id / f"{seq_id}_camera_{cam_id}.txt"

        if in_feature.exists():
            _filter_feature_pkl(in_feature, out_feature, remove_ids)

        if in_metadata.exists():
            _filter_metadata_txt(in_metadata, out_metadata, remove_ids)



if __name__ == "__main__":
    main()