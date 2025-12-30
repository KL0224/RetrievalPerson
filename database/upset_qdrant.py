from mmc.run_matching import run_matching
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid


def upsert_to_qdrant(global_objects, host='localhost', port=6333):
    client = QdrantClient(host=host, port=port)
    collection_name = "person_retrieval"

    # Tạo collection với named vectors
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "reid": VectorParams(size=2048, distance=Distance.COSINE),
                "clip": VectorParams(size=512, distance=Distance.COSINE)
            }
        )
        print(f"Đã tạo collection {collection_name}")

    points = []
    for global_id, data in global_objects.items():
        # Lấy thông tin tóm tắt chung cho global object
        detections = data['metadata']['detections']
        cam_ids = list(set(d['cam_id'] for d in detections))

        # Duyệt qua từng lần xuất hiện (appearance) để tạo point riêng
        for app in data['appearances']:
            # Tạo ID ngẫu nhiên cho mỗi point (vì 1 người có nhiều point)
            point_id = str(uuid.uuid4())

            vectors = {
                "reid": app['vector_reid'].tolist(),
                "clip": app['vector_clip'].tolist()
            }

            # Payload chứa global_id để gom nhóm khi search
            payload = {
                'global_id': global_id,
                'cam_id': app['cam_id'],
                'obj_id': app['obj_id'],
                'related_cams': cam_ids
            }

            points.append(PointStruct(id=point_id, vector=vectors, payload=payload))

    batch_size = 10
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"Đã upsert batch {i} -> {i + len(batch)} points")

    print(f"Hoàn tất upsert tổng cộng {len(points)} points vào {collection_name}")


if __name__ == "__main__":
    camera_files = {
        1: ('data/cam1_meta.txt', 'data/cam1_feat.txt'),
        2: ('data/cam2_meta.txt', 'data/cam2_feat.txt'),
        3: ('data/cam3_meta.txt', 'data/cam3_feat.txt'),
        4: ('data/cam4_meta.txt', 'data/cam4_feat.txt'),
        5: ('data/cam5_meta.txt', 'data/cam5_feat.txt'),
        6: ('data/cam6_meta.txt', 'data/cam6_feat.txt'),
        7: ('data/cam7_meta.txt', 'data/cam7_feat.txt'),
    }
    result = run_matching(camera_files)
    upsert_to_qdrant(result)