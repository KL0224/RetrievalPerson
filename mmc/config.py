import sys
sys.path.append("..")
import json

class MatchingConfig:
    def __init__(self):
        self.similarity_threshold = 0.8 # Ngưỡng similarity để matching object
        self.fps = 30 # FPS of video
        self.reid_dim = 512
        self.clip_dim = 1024
        self.spatial_neighbors = {
            "C1": ["C2", "C3", "C4", "C5", "C7"],
            "C2": ["C1", "C3", "C4", "C5"],
            "C3": ["C1", "C2", "C4", "C5", "C7"],
            "C4": ["C1", "C2", "C3", "C5", "C6"],
            "C5": ["C1", "C2", "C3", "C4", "C6"],
            "C6": ["C4", "C5", "C7"],
            "C7": ["C6", "C4", "C1", "C3"],
        }
        self.temporal_neighbors = {}
        with open("data/metadata/temporal_configs.json", "r") as f:
            self.temporal_neighbors = json.load(f)

config = MatchingConfig()
