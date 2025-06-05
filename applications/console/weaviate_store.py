import time
import weaviate 

from typing import Dict, List

class WeaviateClient:
    def __init__(self, url: str = "http://localhost:8080"):
        self.client = weaviate.Client(url)

        while True:
            try:
                if self.client.is_ready():
                    print("Weaviate connection established. ")
                    break
            except Exception:
                pass
            time.sleep(1)

    def get_classes(self):
        schema = self.client.schema.get()
        return [cls["class"].lower() for cls in schema.get("classes", [])]
    
    def create_class(self, class_name: str):
        if class_name.lower() in self.get_classes():
            print(f"Class '{class_name}' already exists.")
            return True
        try:
            self.client.schema.create_class({
                "class": class_name,
                "vectorizer": "text2vec-transformers",
                "properties": [
                    {"name": "title", "dataType": ["string"]},
                    {"name": "content", "dataType": ["text"]}
                ]
            })
            print(f"Created new class '{class_name}'.")
            return True
        except Exception as e:
            raise e
    
    def get_documents(self, class_name: str):
        try:
            res = self.client.query.get(class_name, ["title"]).do()
            objs = res.get("data", {}).get("Get", {}).get(class_name, [])
            return [obj["title"] for obj in objs]
        except Exception as e:
            raise(f"Failed to fetch existing documents in '{class_name}': {e}")
        
    def upload_documents(self, class_name: str, docs: List[Dict]):
        existing_docs = self.get_documents(class_name=class_name)
        skipped, uploaded = 0, 0
        for doc in docs:
            if doc["title"] in existing_docs:
                print(f"Skipped: `{doc['title']}` already exists in `{class_name}`.")
                skipped += 1
            else:
                try:
                    self.client.data_object.create(doc, class_name)
                    uploaded += 1
                except Exception as e:
                    raise(f"Failed to upload `{doc['title']}`: {e}")
        
        if uploaded:
            print(f"Uploaded {uploaded} new document(s) to '{class_name}'.")
        if skipped:
            print(f"Skipped {skipped} duplicate document(s).")
        print()


    def query_documents(self, query: str, class_name: str, top_k: int = 3) -> List[Dict]:
        try:
            res = self.client.query.get(class_name, ["title", "content"]) \
                    .with_near_text({"concepts": [query], "certainty":0.6 }) \
                    .with_additional(["certainty"]) \
                    .with_limit(top_k) \
                    .do()
            
            docs = res.get("data", {}).get("Get", {}).get(class_name, [])

            if isinstance(docs, dict):
                docs = [docs]
            
            return docs
        except Exception as e:
            print(f"Failed to query documents: {e}")
            return []
