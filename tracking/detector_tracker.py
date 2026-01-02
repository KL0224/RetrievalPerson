from ultralytics import YOLO

def run_tracking(video_path, 
                 vid_stride, 
                 confidence, 
                 save: bool = False, 
                 visualize: bool = False, 
                 project_name: str = None, 
                 name: str = None,
                 show: bool = False,
                 verbose: bool = False):
    model = YOLO("yolov8x.pt")
    try:
        results = model.track(
            source=video_path,
            tracker="botsort.yaml",
            persist=True,
            stream=True,
            classes=[0],
            vid_stride=vid_stride,
            conf=confidence,
            visualize=visualize,
            save=save,
            project=project_name,
            name=name,
            show=show,
            verbose=verbose
        )

        for frame_idx, r in enumerate(results):
            if r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            yield frame_idx, r.orig_img, boxes, ids, confs
    finally:
        del model