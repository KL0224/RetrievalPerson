from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid

def upsert_to_qdrant(global_objects, host='localhost', port=6333, drop=False):
    client = QdrantClient(host=host, port=port)
    collection_name = "person_retrieval"

    # Tạo collection với named vectors
    if drop and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "vector_reid": VectorParams(size=512, distance=Distance.COSINE),
                "vector_clip": VectorParams(size=1024, distance=Distance.COSINE)
            }
        )
        print(f"Đã tạo collection {collection_name}")



    points = []
    for global_id, data in global_objects.items():
        cam_ids = list(set(d['cam_id'] for d in data['appearances']))

        # Duyệt qua từng lần xuất hiện (appearance) để tạo point riêng
        for app in data['appearances']:
            # Tạo ID ngẫu nhiên cho mỗi point (vì 1 người có nhiều point)
            point_id = str(uuid.uuid4())

            vectors = {
                "vector_reid": app['vector_reid'].tolist(),
                "vector_clip": app['vector_clip'].tolist()
            }

            # Payload chứa global_id để gom nhóm khi system_search
            payload = {
                'global_id': global_id,
                'seq_id': app['seq_id'],
                'cam_id': app['cam_id'],
                'obj_id': app['obj_id'],
                'detections': app['detections'],
                'related_cams': cam_ids,
                'frame_start': app['detections'][0]['frame_id'],
                'frame_end': app['detections'][-1]['frame_id']
            }

            points.append(PointStruct(id=point_id, vector=vectors, payload=payload))

    batch_size = 10
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"Đã upsert batch {i} -> {i + len(batch)} points")

    print(f"Hoàn tất upsert tổng cộng {len(points)} points vào {collection_name}")