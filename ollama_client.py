import requests
import json

class OllamaClient:
    def __init__(self, model="codellama", host="http://localhost:11434"):
        self.model = model
        self.host = host
    
    def query(self, prompt):
        """Non-streaming query for simple cases."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(f"{self.host}/api/generate", json=payload)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama error: {response.text}")
    
    def stream_query(self, prompt):
        """Streaming query yielding response chunks."""
        print(f"Streaming query with prompt: {prompt}")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        response = requests.post(f"{self.host}/api/generate", json=payload, stream=True)
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
                if data.get("done", False):
                    break
