class MatchingConfig:
    def __init__(self):
        self.similarity_threshold = 0.8 # Ngưỡng similarity để matching object
        self.spatial_neighbors = {
            "C1": ["C2", "C3"],
            "C2": ["C1", "C3", "C4", "C5"],
            "C3": ["C1", "C2", "C4", "C5"],
            "C4": ["C2", "C3", "C5"],
            "C5": ["C2", "C3", "C4", "C6"],
            "C6": ["C5", "C7"],
            "C7": ["C6"],
        } # Cam lân cận
        self.fps = 30 # FPS of video
        self.reid_dim = 2048
        self.clip_dim = 512

config = MatchingConfig()
