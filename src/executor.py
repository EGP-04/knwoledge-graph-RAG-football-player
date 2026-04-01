# src/executor.py

import json
import re
from src.llm_router import Router
from src.agent import TOOLS
from src.formatter import format_response
from src.node_extraction import extractor
from toolkit.retriever_tools import get_retriever_tool_context

router = Router()

def decide_tool(query):
    """
    LLM decides which tool to use + extracts parameters
    """

    tool_context = get_retriever_tool_context()
    prompt = f"""
    You are a strict JSON generator.

    You MUST return ONLY a valid JSON object.
    Do NOT include:
    - explanations
    - text before or after JSON
    - markdown
    - code blocks

    If you violate this, the output is invalid.

    {tool_context}

    Goal:
    - Retrieve the most relevant information for the user's query
    - Support MULTI-HOP reasoning using multiple tool calls
    - Later tool calls MUST use outputs from earlier calls when needed

    Output format:
    {{
        "calls": [
            {{
                "id": "step1",
                "tool": "tool_name",
                "params": {{
                    "param_name": "value"
                }}
            }},
            {{
                "id": "step2",
                "tool": "tool_name",
                "params": {{
                    "param_name": "$step1.output_field"
                }}
            }}
        ]
    }}

    Rules:
    - You may return ONE or MORE tool calls
    - Each call MUST have a unique "id"
    - If a step depends on a previous step, reference it using:
    "$<step_id>.<field>"
    - Do NOT invent fields — only use fields that the tool would return
    - Maintain correct execution order
    - Use tool names exactly as given
    - Include ONLY relevant parameters

    Multi-hop strategy:
    1. Break the query into steps
    2. Extract entities using first call(s)
    3. Use outputs (IDs, names, relations) in later calls
    4. Chain calls logically

    Example:
    If step1 returns:
    {{ "player_id": 123 }}

    Then step2 can use:
    "$step1.player_id"

    User question:
    {query}
    ANSWER ONLY BASED ON THE QUERY
    ONLY OUTPUT JSON.
    """
    response = router.generate(prompt)
    print("ROUTER RESPONSE: ",response)
    return json.loads(response)


def execute(query):
    """
    Full pipeline:
    1. Decide tools (multi-hop)
    2. Resolve and execute tool calls sequentially
    3. Format overall response using results
    """

    decision = decide_tool(query)
    calls = decision.get("calls", [])
    
    if not calls:
        # Compatibility with old single-tool format if still returned by LLM
        tool_name = decision.get("tool")
        params = decision.get("params", {})
        if tool_name:
            calls = [{"id": "step1", "tool": tool_name, "params": params}]
        else:
            # Maybe it returned a raw dict with calls inside from another format
            if isinstance(decision, dict) and "calls" not in decision:
                # If it's just one call directly
                pass 
            return format_response(query, [])

    results_map = {}
    all_context = []

    for call in calls:
        step_id = call.get("id", f"step_{len(results_map)}")
        tool_name = call.get("tool")
        raw_params = call.get("params", {})

        # Resolve parameter dependencies ($step_id.field)
        resolved_params = {}
        for k, v in raw_params.items():
            if isinstance(v, str) and v.startswith("$"):
                try:
                    # Format: $step_id.field
                    if "." in v:
                        ref_id, field = v[1:].split(".", 1)
                        prev_out = results_map.get(ref_id)
                        
                        if field == "output":
                            resolved_params[k] = prev_out
                        elif prev_out and isinstance(prev_out, list) and len(prev_out) > 0:
                            # Typically our tools return a list of dicts. Pick the first one.
                            resolved_params[k] = prev_out[0].get(field)
                        elif prev_out and isinstance(prev_out, dict):
                            resolved_params[k] = prev_out.get(field)
                        else:
                            resolved_params[k] = None
                    else:
                        # Fallback if no dot: use the whole previous result
                        resolved_params[k] = results_map.get(v[1:])
                except (ValueError, AttributeError):
                    resolved_params[k] = v
            else:
                resolved_params[k] = v

        print(f"Executing {step_id}: {tool_name} with {resolved_params}")

        if tool_name in TOOLS:
            try:
                # Semantic entity resolution
                resolved_params = extractor.resolve_params(tool_name, resolved_params)
                print(f"[DEBUG] Final resolved params for {step_id}: {resolved_params}")

                out = TOOLS[tool_name](**resolved_params)
                results_map[step_id] = out
                all_context.append({
                    "step": step_id,
                    "tool": tool_name,
                    "params": resolved_params,
                    "output": out
                })
            except Exception as e:
                print(f"[ERROR] Step {step_id} ({tool_name}) failed: {e}")
                results_map[step_id] = None
        else:
            print(f"[WARNING] Tool {tool_name} not found")
            results_map[step_id] = None

    # Pass the entire chain context to the formatter
    return format_response(query, all_context)