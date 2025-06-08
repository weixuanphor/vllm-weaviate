import logging
import os
import requests

from pathlib import Path

from config import API_URL, MODEL_NAME, MAX_TOKENS, TEMPERATURE, TOP_P, TOP_K
from doc_reader import DocumentReader
from weaviate_store import WeaviateClient

class LLMClient:
    def __init__(self, weaviate_client, api_url=API_URL, model_name=MODEL_NAME, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P):
        self.weaviate_client = weaviate_client
        self.api_url = api_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def build_prompt(self, question, history=None, docs=None):
        history = history or []
        docs = docs or []
        parts = []

        if docs:
            docs_text = "\n\n".join([f"{d['title']}:\n{d['content']}" for d in docs])
            parts.append(f"Context documents:\n{docs_text}")

        if history:
            history_text = ""
            for user_msg, assistant_msg in history:
                history_text += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
            parts.append(f"Chat history:\n{history_text.strip()}")

        parts.append(f"Answer the user question:\n{question}")
        prompt = "\n\n".join(parts)
        prompt += "\n\nPlease answer concisely and directly, you may provide some explanations if suitable. Your final answer shouldn't contain any internal instructions."

        return prompt

    def generate_response(self, query, class_name="", history=None, enable_rag=False):
        if history is None:
            history = []

        docs = []
        if enable_rag:
            try:
                docs = self.weaviate_client.query_documents(query=query, class_name=class_name, top_k=TOP_K)
                references = []
                for d in docs:
                    references.append(d["title"])
                logging.info(f"{docs}")
            except Exception as e:
                print(f"Warning: Failed to fetch documents: {e}")

        prompt = self.build_prompt(query, history, docs)
        print(f"Prompt: {prompt}")
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "stream": False,
        }

        response = requests.post(API_URL, json=data)
        response.raise_for_status()
        answer = response.json()["choices"][0]["text"]

        if references:
            answer += f"\nReferences: {str(references)}\n"
        return answer

    def extract_answer(self, text: str) -> str:
        # Split text by separator lines (---)
        chunks = [chunk.strip() for chunk in text.split('---')]
        
        # Take the first non-empty chunk
        for chunk in chunks:
            if chunk:
                first_chunk = chunk
                break
        else:
            # If all chunks empty, return original stripped text
            first_chunk = text.strip()

        # Handle dash-prefixed answers in that chunk
        lines = first_chunk.splitlines()
        if lines and lines[0].startswith("-"):
            # Remove leading dash and whitespace from each line
            cleaned_lines = [line.lstrip("- ").rstrip() for line in lines]
            return "\n".join(cleaned_lines).strip()

        return first_chunk


if __name__ == "__main__":
    conversation_history = []
    doc_paths = [
        "/home/amd/weixphor/vllm-weaviate/assets/advancing-ai-2025-distribution-deck.pdf", 
        "/home/amd/weixphor/vllm-weaviate/assets/Data and Model Poisoning.txt",
        "/home/amd/weixphor/vllm-weaviate/assets/doc1.txt",
        "/home/amd/weixphor/vllm-weaviate/assets/doc2.txt",
        "/home/amd/weixphor/vllm-weaviate/assets/doc3.txt"
    ]
    docs_to_upload = []
    class_name = "Test_pdf_txt" # weaviate will capitalize first letter

    # Initialize required instances
    weaviate_client = WeaviateClient()
    llm_client = LLMClient(weaviate_client)
    doc_reader = DocumentReader()
    
    # Upload docs
    if doc_paths:
        for doc_path in doc_paths:
            docs_to_upload.append(doc_reader.read_document(Path(doc_path)))    

    if docs_to_upload:
        created = weaviate_client.create_class(class_name)
        if created:
            weaviate_client.upload_documents(class_name, docs_to_upload)

    # Set enable_rag
    if weaviate_client.get_classes():
        enable_rag = True
        print("RAG enabled. Classes available:", weaviate_client.get_classes())
        print("Documents available:", weaviate_client.get_documents(class_name=class_name))
        print()

    # For logging conversations
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/conversation.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w" 
    )

    idx = 0
    while True:
        print(f"Iteration #{idx+1}")
        q = input("You: ")
        if q.lower() == "exit":
            break

        response = llm_client.generate_response(query=q, class_name=class_name, history=conversation_history, enable_rag=enable_rag)
        answer = llm_client.extract_answer(response)
        print()
        print("Assistant:", answer)
        print("*"*30)
        conversation_history.append((q, answer))

        # Log the interaction
        logging.info(f"Iteration #{idx+1}")
        logging.info(f"You: {q}")
        logging.info(f"Assistant: {answer}")
        logging.info("*" * 30)

        idx += 1



# Required (one of the following):
# vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
# vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
# https://huggingface.co/deepseek-ai/DeepSeek-R1 
# vllm serve Qwen/Qwen2.5-7B-Instruct
# vllm serve mistralai/Mistral-7B-Instruct-v0.3  --tensor-parallel-size 2 --max-model-len 32768 --enforce-eager
# vllm serve meta-llama/Llama-3.2-3B-Instruct

# Required:
# docker-compose up weaviate