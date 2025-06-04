# Requires vllm serve
#  

import requests

API_URL = "http://localhost:8000/v1/completions"
MODELS = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"]
OPTION = 0

def generate_in_chunks(
    prompt: str,
    model: str,
    max_tokens_per_call: int = 100,
    max_total_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop_condition=None  # Optional: function that accepts generated_text and returns True to stop
) -> str:
    """
    Generate text from the model in multiple calls, appending previous output to prompt.

    Args:
      prompt: Initial prompt string.
      model: Model name string.
      max_tokens_per_call: Tokens to generate per request.
      max_total_tokens: Max tokens to generate overall.
      temperature: Sampling temperature.
      top_p: Nucleus sampling parameter.
      stop_condition: Optional function(generated_text) -> bool to stop early.

    Returns:
      The full generated text.
    """
    generated_text = ""
    current_prompt = prompt
    tokens_generated = 0

    while tokens_generated < max_total_tokens:
        data = {
            "model": model,
            "prompt": current_prompt,
            "max_tokens": max_tokens_per_call,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }

        response = requests.post(API_URL, json=data)
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code} {response.text}")

        result = response.json()
        new_text = result['choices'][0]['text']

        generated_text += new_text
        tokens_generated += max_tokens_per_call  # Rough estimate

        # Update prompt to include all generated text so far
        current_prompt = prompt + generated_text

        if stop_condition and stop_condition(generated_text):
            break

    return generated_text


# Stop condition - stop if last char is a period.
def stop_on_period(text):
    return text.strip().endswith(".")

if __name__ == "__main__":
    prompt = "Once upon a time, in a faraway kingdom,"
    story = generate_in_chunks(prompt, MODELS[OPTION], max_tokens_per_call=150, max_total_tokens=600, temperature=0.3, top_p=0.95, stop_condition=stop_on_period)
    print("Generated text:\n", story)
