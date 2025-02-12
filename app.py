import os
import numpy as np
import face_recognition
from elasticsearch import Elasticsearch

# تنظیمات اتصال به Elasticsearch در Docker
ELASTIC_URL = "http://localhost:9200"
INDEX_NAME = "face_index"
es = Elasticsearch(ELASTIC_URL, headers={"Content-Type": "application/json"})


# ایجاد ایندکس برای جستجوی k-NN بدون نیاز به elastiknn
def create_index():
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    
    settings = {
        "settings": {"number_of_shards": 1},
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": 128  # تعداد ابعاد خروجی face_recognition
                },
                "name": {"type": "keyword"}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, body=settings)

# پردازش تصاویر و ذخیره Embeddings
def index_faces(image_dir):
    for file_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, file_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            embedding = encodings[0].tolist()
            doc = {"embedding": embedding, "name": file_name}
            es.index(index=INDEX_NAME, body=doc)

# جستجوی تصویر مشابه با استفاده از cosine similarity
def search_face(image_path, k=5):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    
    if len(encodings) == 0:
        return "No face detected"
    
    query = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": encodings[0].tolist()}
                }
            }
        }
    }
    
    results = es.search(index=INDEX_NAME, body=query)
    return [hit["_source"]["name"] for hit in results["hits"]["hits"]]

# اجرای مراحل پروژه
if __name__ == "__main__":
    create_index()
    index_faces("00000")  # مسیر دیتاست
    search_results = search_face("/Users/haniehhosseini/ir/00000.png")
    print("نتایج جستجو:", search_results)
