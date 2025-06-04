import requests

url = "http://localhost:8000/v1/completions"

data = {
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "prompt": "What is 1+1?",
    "max_tokens": 200,
}

response = requests.post(url, json=data)

if response.status_code == 200:
    completion = response.json()
    print("Model response:", completion["choices"][0]["text"])
else:
    print("Request failed:", response.status_code, response.text)

