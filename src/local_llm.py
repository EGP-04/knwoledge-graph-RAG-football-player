import requests

class LocalLLM:
    def __init__(self):
        self.url = "http://localhost:1234/v1/chat/completions"

    def generate(self, prompt: str):
        payload = {
            "model": "qwen/qwen3-4b-thinking-2507",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        response = requests.post(self.url, json=payload)
        return response.json()["choices"][0]["message"]["content"]
