from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from deep_person_reid.torchreid.utils.feature_extractor import FeatureExtractor

def run_tracking(video_path, 
                 vid_stride, 
                 confidence, 
                 model_name = 'yolov8x.pt',
                 save: bool = False, 
                 visualize: bool = False, 
                 project_name: str = None, 
                 name: str = None,
                 show: bool = False,
                 verbose: bool = False,
                 device: str = 'cpu'):
    # Initialize model and tracker
    model = YOLO(model_name)

    embedder = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='./models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth',
        device= device,
    )

    tracker = DeepSort(max_age=30) # must define tracker here
    
    try:
        # Detecting
        results = model(
            source=video_path,
            stream=True,
            classes=[0],
            vid_stride=vid_stride,
            conf=confidence,
            visualize=visualize,
            save=save,
            project=project_name,
            name=name,
            show=show,
            verbose=verbose,
            device=device
        )

        # Tracking
        for frame_idx, r in enumerate(results):
            if r.boxes is None:
                continue
            frame = r.orig_img # HxWxC, numpy array
            bbs = [] # List[ Tuple( List[float or int], float, str ) ] ( [left,top,w,h] , confidence, detection_class)
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            # boxes_xywh = r.boxes.xywh.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            crops = []
            for box, conf in zip(boxes_xyxy, confs):
                left, top, right, bottom = map(int, box)
                w = right - left
                h = bottom - top
                crop = frame[top:bottom, left:right]
                crops.append(crop)
                bbs.append( ([left, top, w, h], conf, '0') )
            
            embeds = embedder(crops) # your own embedder to take in the cropped object chips, and output feature vectors
            tracks = tracker.update_tracks(bbs, embeds)
            ids = []
            confs = []
            ltrb_boxes = []
            # ltwh_boxes = []
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = int(track.track_id)
                ltrb = track.to_ltrb(orig=True, orig_strict=True)
                if ltrb is None:
                    ltrb = [0,0,0,0]
                # ltwh = track.to_ltwh(orig=True, orig_strict=True)
                conf = track.get_det_conf()
                
                if conf is None:
                    conf = 0.0
            
                ids.append(track_id)
                confs.append(conf)
                ltrb_boxes.append(ltrb)
                # ltwh_boxes.append(ltwh)
            print(ltrb_boxes)
            boxes_xyxy = np.array(ltrb_boxes)
            # boxes_xywh = np.array(ltwh_boxes)
            ids = np.array(ids)
            confs = np.array(confs)
            yield frame_idx, r.orig_img, boxes_xyxy, ids, confs
    finally:
        del model