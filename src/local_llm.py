import requests

class LocalLLM:
    def __init__(self):
        self.url = "http://127.0.0.1:1234/v1/chat/completions"

    def generate(self, prompt: str):
        payload = {
            "model": "qwen/qwen3-4b",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        try:
            response = requests.post(self.url, json=payload)
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            elif "error" in data:
                return f"[LocalLLM Error] {data['error']}"
            else:
                return f"[LocalLLM Error] Unexpected response: {data}"
        except requests.exceptions.RequestException as e:
            return f"[LocalLLM Connection Error] Could not reach the LLM at {self.url}. Is it running? Details: {e}"
