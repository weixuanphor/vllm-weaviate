import weaviate
import time

client = weaviate.Client("http://localhost:8080")

# Wait until weaviate is ready
while True:
    try:
        if client.is_ready():
            break
    except Exception:
        pass
    print("Waiting for Weaviate to be ready...")
    time.sleep(2)

schema = client.schema.get()
doc_name = "Documen"
doc_exists = 0
for doc in schema["classes"]:
    if doc_name == doc["class"]:
        doc_exists = 1
        break

if doc_exists:
    print(f"Document named {doc_name} already exists.")
else:
    # Create schema
    client.schema.create_class({
        "class": doc_name,
        "vectorizer": "text2vec-transformers",
        "properties": [
            {"name": "title", "dataType": ["string"]},
            {"name": "content", "dataType": ["text"]}
        ]
    })
    print(f"Created new document named {doc_name}.")

# Add sample documents
docs = [
    {"title": "Doc 1", "content": "This document is about AI and machine learning."},
    {"title": "Doc 2", "content": "Weaviate provides semantic vector search."},
    {"title": "Doc 3", "content": "This is a paper about cats."}
]

for doc in docs:
    client.data_object.create(doc, doc_name)

# Query example
query_text = "What is the paper about?"
result = client.query.get(doc_name, ["title", "content"]) \
    .with_near_text({"concepts": [query_text]}) \
    .with_limit(2) \
    .do()

print("Query results:")
print(result)
