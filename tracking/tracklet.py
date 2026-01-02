from dataclasses import dataclass
import numpy as np

@dataclass
class TrackletFrame:
    frame_id: int
    bbox: np.ndarray
    confidence: float
    image: np.ndarray


class Tracklet:
    def __init__(self, global_id, camera_id):
        self.global_id = global_id
        self.camera_id = camera_id
        self.frames = []
        self.reid_embeddings = []
        self.clip_embeddings = []

    def add_frame(self, frame_id, bbox, confidence, image):
        self.frames.append(
            TrackletFrame(frame_id, bbox, confidence, image)
        )


class TrackletManager:
    def __init__(self):
        self.tracklets = {}

    def get(self, global_id, camera_id):
        if global_id not in self.tracklets:
            self.tracklets[global_id] = Tracklet(global_id, camera_id)
        return self.tracklets[global_id]

    def all(self):
        return self.tracklets.values()
