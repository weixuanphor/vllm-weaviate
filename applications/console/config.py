# config.py

# LLM API endpoint
API_URL = "http://localhost:8000/v1/completions"

# Weaviate client URL
WEAVIATE_URL = "http://localhost:8080"

# Default model for generation
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Embedding model name or config (if you have one)
EMBEDDING_MODEL = "your-embedding-model-name-or-id"

# Number of top documents to retrieve from Weaviate
TOP_K = 3

# Other config constants (timeouts, max tokens, etc.)
MAX_TOKENS = 600
TEMPERATURE = 0.4
TOP_P = 0.9
