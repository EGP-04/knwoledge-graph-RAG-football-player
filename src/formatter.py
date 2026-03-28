from src.local_llm import LocalLLM

small_model = LocalLLM()


def format_response(user_query: str, data):
    print("RETRIEVED DATA: ",data)
    """
    Converts graph output into a natural language answer
    using both:
    - user query
    - retrieved data (context)
    """

    prompt = f"""
    You are a football assistant.

    User Question:
    {user_query}

    Retrieved Data:
    {data}

    Instructions:
    - Answer ONLY using the retrieved data
    - Do NOT hallucinate
    - If data is empty, say "I don't know"
    - Keep the answer concise and clear

    Final Answer:
    """

    return small_model.generate(prompt)