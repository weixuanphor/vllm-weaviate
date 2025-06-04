from vllm import LLM
import torch.distributed as dist
import atexit

@atexit.register
def shutdown_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    MODEL_PATH = "/home/amd/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/711ad2ea6aa40cfca18895e8aca02ab92df1a746/"

    print(f"Loading model from: {MODEL_PATH}")
    llm = LLM(model=MODEL_PATH, tokenizer=None, tensor_parallel_size=2)

    prompt = "Hello, how are you?"
    outputs = llm.generate([
        {
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    ])
    result = outputs[0].outputs[0].text
    print("\nGenerated response:\n", result)

if __name__ == "__main__":
    main()
