from flask import Flask, render_template, jsonify, send_from_directory, request, url_for
from system_search.search import SystemSearch
from werkzeug.utils import secure_filename
import os
import re
import numpy as np
import subprocess

# Init app and system
app = Flask(__name__, template_folder='templates', static_folder='static')
USE_GPU = True


# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", 'uploads') # Thư mục upload ảnh
VIDEO_FOLDER = os.path.join(BASE_DIR, "static", 'videos') # Thư mục chứa video
VIDEO_CROP_FOLDER = os.path.join(BASE_DIR, "static", 'video_crop') # Thư mục chứa video crop cho từng đối tượng
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(VIDEO_CROP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB cho ảnh

# Utils
system_search = None
def get_search_system():
    global system_search
    if system_search is None:
        system_search = SystemSearch()
    return system_search

system_search = get_search_system()

def video_input_path(seq_id: str, cam_id: str) -> str:
    return os.path.join(VIDEO_FOLDER, str(seq_id), f'{cam_id}.avi')

def _safe_str(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(s))

def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))

def _bbox_to_xyxy(bbox):
    if not bbox or len(bbox) != 4:
        raise ValueError('bbox must have 4 elements')

    a, b, c, d = map(int, bbox)
    if c < a or d < b:
        a, b, c, d = a, b, a + c, b + d

    return a, b, c, d

def clamp_bbox_xyxy(b, w, h):
    x1, y1, x2, y2 = _bbox_to_xyxy(b)
    x1 = _clamp(int(x1), 0, w - 1)
    y1 = _clamp(int(y1), 0, h - 1)
    x2 = _clamp(int(x2), 0, w - 1)
    y2 = _clamp(int(y2), 0, h - 1)
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

# ffmpeg helpers
VIDEO_METADATA = {
    'w': 1080,
    'h': 720,
    'fps': 30
}


def ffmpeg_decode_segment(video_path: str, ss: float, t: float):
    cmd = [
        "ffmpeg",
        "-ss", f"{ss:.6f}",
        "-t", f"{t:.6f}",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def ffmpeg_encode(output_path:str, fps:int, out_w: int, out_h: int):
    """Encode video frames using ffmpeg"""
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_path
    ]

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

# Page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/details")
def details():
    return render_template("detail.html")

# API
@app.route("/api/search", methods=["POST"])
def api_search():
    text_query = (request.form.get("query") or "").strip()
    file = request.files.get("file")

    if not text_query and (file is None or file.filename == ""):
        return jsonify({"error": "No text query or image file provided"}), 400

    image_path = None
    save_path = None

    if file and file.filename:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        image_path = save_path

    try:
        results = system_search.search(
            image_path=image_path,
            text_query=text_query,
            max_results=100
        ) or []

        payload = []
        for r in results:
            payload.append({
                "global_id": r.get("global_id"),
                "score": r.get("score", 0.0),
                "thum_url": r.get("thum_url"),
                "cameras": r.get("cameras", []),
                "tracks": r.get("tracks", []),
                "detail_url": url_for("details", _external=False)
            })

        return jsonify(payload)

    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

@app.route("/api/get_video", methods=["POST"])
def api_get_video():
    data = request.get_json(force=True)

    global_id = data.get("global_id")
    seq_id = data.get("seq_id")
    cam_id = data.get("cam_id")
    obj_id = data.get("obj_id")
    tracks = data.get("tracks")

    required = ["global_id", "seq_id", "cam_id", "obj_id"]
    if any(data.get(k) is None for k in required) or not tracks:
        return jsonify({"error": "Missing required fields"}), 400

    track = next(
        (t for t in tracks
         if str(t.get("seq_id")) == str(seq_id)
         and str(t.get("cam_id")) == str(cam_id)
         and str(t.get("obj_id")) == str(obj_id)),
        None
    )

    if not track:
        return jsonify({"error": "No track found"}), 400

    detections = track.get("detections")
    if not detections:
        return jsonify({"error": "No detections found"}), 400
    try:
        detections_sorted = sorted(detections, key=lambda d: int(d.get("frame_id", 0)))
        det_map = {
            int(d["frame_id"]): d["bbox"]
            for d in detections_sorted
            if "frame_id" in d and "bbox" in d and d["bbox"] is not None
        }
    except Exception as e:
        return jsonify({"error": f"Invalid detections format with error {e}"}), 400

    if not det_map:
        return jsonify({"error": "Detections empty after parsing"}), 400

    start_f = min(det_map)
    end_f = max(det_map)

    video_crop_name = (
        f"{_safe_str(global_id)}__seq{seq_id}__cam{cam_id}__obj{obj_id}.mp4"
    )

    cache_path = os.path.join(VIDEO_CROP_FOLDER, video_crop_name)

    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return jsonify({
            "video_url": url_for("static", filename=f"video_crop/{video_crop_name}", _external=False),
            "cached": True,
            "start_frame": start_f,
            "end_frame": end_f,
            "fps": VIDEO_METADATA.get("fps", 30),
        })

    # Crop video with ffmpeg
    video_path = video_input_path(seq_id, cam_id)
    if not os.path.exists(video_path):
        return jsonify({"error": "No video found"}), 400

    w = int(VIDEO_METADATA.get("w", 1080))
    h = int(VIDEO_METADATA.get("h", 720))
    fps = int(VIDEO_METADATA.get("fps", 30))

    start_time = start_f / fps
    duration = (end_f - start_f + 1) / fps

    skip_rate = 1
    if len(det_map) > 1000:
        skip_rate = 10
    elif len(det_map) > 300:
        skip_rate = 5

    draw_filters = []
    for f_c, (frame_id, bbox) in enumerate(det_map.items()):
        if f_c % skip_rate != 0:
            continue

        local_n = int(frame_id) - int(start_f)
        if local_n < 0:
            continue
        try:
            x1, y1, x2, y2 = clamp_bbox_xyxy(bbox, w=w, h=h)
        except Exception as e:
            continue

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Thêm câu lệnh vẽ box theo từng frame
        draw_filters.append(
            f"drawbox=x={x1}:y={y1}:w={bw}:h={bh}:color=lime@1.0:t=3:enable='eq(n\\,{local_n})'"
        )

    # Tổng hợp lệnh vẽ
    vf = ",".join(draw_filters) if draw_filters else "null"

    if USE_GPU:
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-ss", f"{start_time:.4f}",
            "-t", f"{duration:.4f}",
            "-i", video_path,
            "-vf", vf,
            "-c:v", "h264_nvenc",
            "-preset", "p1",
            "-pix_fmt", "yuv420p",
            "-an",
            cache_path
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_time:.6f}",
            "-t", f"{duration:.6f}",
            "-i", video_path,
            "-vf", vf,
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            cache_path,
        ]

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            err = (p.stderr or "").strip()
            return jsonify({"error": "FFmpeg failed", "details": err[-2000:]}), 500
    except FileNotFoundError:
        return jsonify({"error": "ffmpeg not found on PATH"}), 500

    if not os.path.exists(cache_path) or os.path.getsize(cache_path) == 0:
        return jsonify({"error": "Failed to generate annotated clip"}), 500

    return jsonify({
        "video_url": url_for("static", filename=f"video_crop/{video_crop_name}", _external=False),
        "cached": False,
        "start_frame": start_f,
        "end_frame": end_f,
        "fps": fps,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)