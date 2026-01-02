SEQUENCES = {
    0: 'seq_001',
    1: 'seq_002',
    2: 'seq_004',
    3: 'seq_006',
    4: 'seq_007',
    5: 'seq_020',
    6: 'seq_024',
    7: 'seq_025',
    8: 'seq_026'
}


CAMERAS = {
    # 0: "videos/cam1.mp4",
    # 1: "videos/cam2.mp4",
    # 2: "videos/cam3.mp4",
    3: "videos/test_clip.mp4",  #"videos/seq_4camera_3_2023-06-08-12:29:08.mp4",
    # 4: "videos/cam5.mp4",
    # 5: "videos/cam6.mp4",
    # 6: "videos/cam7.mp4",
}

REAL_FPS = 30
VID_STRIDE = 3                 # or 3, 4
VIRTUAL_FPS = REAL_FPS // VID_STRIDE # = 10
CONFIDENCE_THRESHOLD = 0.3

SAMPLE_EVERY_FRAMES = 10       # 1 second
REID_WINDOW_SECONDS = 3       # aggregate every 3 seconds

SEQ_ID_OFFSET = 1_000_000
CAMERA_ID_OFFSET = 100_000

#models
REID_MODEL_NAME = "osnet_x1_0"
REID_EMB_SIZE = 512

CLIP_MODEL_NAME = "ViT-H-14-378-quickgelu"
CLIP_EMB_SIZE = 1024