from src.local_llm import LocalLLM

small_model = LocalLLM()


def format_response(user_query: str, data):
    print("RETRIEVED DATA: ", data)
    """
    Converts graph output into a natural language answer
    using both:
    - user query
    - retrieved data (context)
    """

    # Extract only the output of the last step for the LLM context
    last_step_output = []
    if isinstance(data, list) and len(data) > 0:
        last_step_output = data[-1].get("output", [])
    else:
        last_step_output = data

    prompt = f"""
    You are a football assistant.

    User Question: 
    {last_step_output}

    
    Retrieved Data:
    {last_step_output}
    These are the answer field convert them to a sentence based on the question. 
    The answer should be formulated in such a way that it is the answer to the question
    For example if the question is "Which club does M. Neuer play for?", Answer as he plays for club_name.
    Final Answer:
    """

    return small_model.generate(prompt)