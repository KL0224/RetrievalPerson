import torch
from system_search.model import ReIDModel, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from logger import get_logger
from PIL import Image
import numpy as np
from collections import defaultdict

logger = get_logger("search")

class SystemSearch:
    def __init__(self):
        logger.info("Initializing SystemSearch")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.reidmodel = ReIDModel(device=self.device, model_path='models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth')
        logger.info("Initialized ReIDModel")

        self.clipmodel = CLIPModel(self.device)
        logger.info("Initialized CLIPModel")

        self.client = QdrantClient(host="localhost", port=6333)
        logger.info("Initialized QdrantClient")
        self.collection_name = "person_retrieval"

    # system_search/search.py
    def parse_qdrant_outputs(self, objects):
        results = {}

        for group in objects.groups:
            global_id = group.id
            hit = group.hits[0]
            score = hit.score

            payload = hit.payload or {}
            cam_id = payload["cam_id"]
            seq_id = payload["seq_id"]
            obj_id = payload["obj_id"]

            results[global_id] = {
                "score": score,
                "seq_id": seq_id,
                "cam_id": cam_id,
                "obj_id": obj_id,
            }

        return results

    def filter_global_id(self, global_id_dict):
        objects_full_list = defaultdict(list)
        id_list = list(global_id_dict.keys())

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="global_id",
                        match=models.MatchAny(any=id_list),
                    )
                ]
            ),
            limit=200
        )

        for point in points:
            payload = point.payload
            global_id = payload["global_id"]

            objects_full_list[global_id].append({
                "point_id": point.id,
                "seq_id": payload["seq_id"],
                "cam_id": payload["cam_id"],
                "obj_id": payload["obj_id"],
                "detections": payload["detections"],
                "related_cams": payload["related_cams"],
                "frame_start": payload["frame_start"],
                "frame_end": payload["frame_end"],
            })

        return objects_full_list

    def search(self, image_path=None, text_query=None, max_results=10):
        """
        Search theo 3 trường hợp: chỉ có ảnh, chỉ có text hoặc có cả 2
        :param image_path: đường dẫn tới ảnh
        :param text_query: string của query
        :param max_results: số lượng kết quả lớn nhất
        :return: max_results dict cho kết quả
        """
        try:
            # Trường hợp chỉ có text không có image
            if text_query and image_path is None:
                emb_text = self.clipmodel.encode_text(text_query)
                objects =  self.search_text_only(emb_text, max_results)
            elif image_path and text_query is None:  # Trường hợp chỉ có image
                img = Image.open(image_path).convert('RGB')
                img_np = np.array(img)
                emb_img_reid = self.reidmodel.extract(img_np)[0]
                emb_img_clip = self.clipmodel.encode_image(img_np)
                objects = self.search_image_only(emb_img_reid, emb_img_clip, max_results)
            else: # Có cả 2
                emb_text = self.clipmodel.encode_text(text_query)
                img = Image.open(image_path).convert('RGB')
                img_np = np.array(img)
                emb_img_reid = self.reidmodel.extract(img_np)[0]
                objects =  self.search_hybrid(emb_img_reid, emb_text, max_results)

            object_dict =  self.parse_qdrant_outputs(objects)

            if not object_dict:
                return []

            global_details = self.filter_global_id(object_dict)

            # Lấy thông tin chi tiết của từng global_id cho frontend
            final_results = []
            for gid, info in object_dict.items():
                if gid in global_details:
                    tracks = global_details[gid]
                    cam_ids = list(set(t["cam_id"] for t in tracks)) # Lấy danh sách cam đi qua của đối tượng đó
                    cam_ids.sort()

                    # Hình đại diện
                    thum_url = f"static/crops/{info['seq_id']}/{info['cam_id']}/{info['obj_id']}.jpg"

                    final_results.append({
                        "global_id": gid,
                        "score": info["score"],
                        "cameras": cam_ids,
                        "thum_url": thum_url,
                        "tracks": tracks,
                    })

            return final_results
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def search_text_only(self, vector_text, limit=10):
        return self.client.query_points_groups(
            collection_name=self.collection_name,
            query=vector_text,
            using="vector_clip",
            group_by="global_id",
            group_size=1,
            limit=limit,
            with_payload=True
        )

    def search_image_only(self, vector_reid, vector_clip_image, limit=10):
        return self.client.query_points_groups(
            collection_name=self.collection_name,
            prefetch= [
                models.Prefetch(
                    query=vector_clip_image,
                    using="vector_clip",
                    limit= limit * 3
                ),

                models.Prefetch(
                    query=vector_reid,
                    using="vector_reid",
                    limit= limit * 3
                ),
            ],
            query= models.FusionQuery(fusion=models.Fusion.RRF),
            limit= limit,
            with_payload=True,
            group_by="global_id",
            group_size=1,
        )

    def search_hybrid(self, vector_reid, vector_clip_text, limit=10):
        return self.client.query_points_groups(
            collection_name=self.collection_name,
            prefetch= [
                models.Prefetch(
                    query=vector_clip_text,
                    using="vector_clip",
                    limit= limit * 100
                )
            ],
            query = vector_reid,
            using="vector_reid",
            limit= limit,
            with_payload=True,
            group_by="global_id",
            group_size=1,
        )


