from src.local_llm import LocalLLM

class GeminiRouter:
    """
    Router that uses the local LLM hosted via LM Studio.
    Renamed/kept for compatibility with existing imports.
    """
    def __init__(self):
        self.model = LocalLLM()

    def generate(self, prompt):
        return self.model.generate(prompt)

