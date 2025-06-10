import weaviate

# Connect to your Weaviate instance
client = weaviate.Client("http://localhost:8080")

# Define a simple schema for testing
class_obj = {
    "class": "TestDocument",
    "description": "A test class for documents",
    "properties": [
        {
            "name": "text",
            "dataType": ["text"],
            "description": "The document text"
        }
    ]
}

# Create schema if not exists
if not client.schema.contains({"class": "TestDocument"}):
    client.schema.create_class(class_obj)
    print("Schema created.")
else:
    print("Schema already exists.")

# Add sample data objects
sample_texts = [
    "Hello world, this is a test document.",
    "Weaviate is a vector search engine.",
    "You can store and search documents here."
]

for text in sample_texts:
    client.data_object.create(
        data_object={"text": text},
        class_name="TestDocument"
    )
print("Sample data objects added.")

# Query by keyword (full text search)
result = (
    client.query
    .get("TestDocument", ["text"])
    .with_where({
        "path": ["text"],
        "operator": "Like",
        "valueText": "%vector%"
    })
    .do()
)

print("Query results:")
for item in result["data"]["Get"]["TestDocument"]:
    print("-", item["text"])
