import streamlit as st
import os
import sys
import json

# Ensure project root is in sys.path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.executor import execute_with_trace

st.set_page_config(page_title="Football RAG Visualizer", layout="wide")

st.title("⚽ Knowledge-Graph RAG Visualizer")
st.markdown("Ask a question about football players, clubs, leagues, or positions. This tool shows the **invisible reasoning steps** the LLM takes to fetch your answer.")

query = st.text_input("Enter your question:", placeholder="e.g. Which Spanish striker plays for Real Madrid?")

if st.button("Query") and query:
    st.divider()
    
    # Placeholders for dynamic UI updates
    status_text = st.empty()
    
    # UI Containers for each stage
    router_expander = st.expander("🤖 1. Router LLM Plan", expanded=True)
    extraction_expander = st.expander("🧹 2. Entity Resolution & Node Extraction", expanded=False)
    graph_expander = st.expander("📊 3. Graph Database Extraction", expanded=False)
    final_container = st.container()
    
    has_extraction = False
    has_graph_data = False

    with st.spinner("Processing..."):
        try:
            for trace in execute_with_trace(query):
                t_type = trace.get("type")
                
                if t_type == "status":
                    status_text.info(f"🔄 **Status:** {trace.get('data')}")
                    
                elif t_type == "router_decision":
                    decision = trace.get("data")
                    with router_expander:
                        st.json(decision)
                        
                elif t_type == "node_extraction":
                    has_extraction = True
                    step_id = trace.get("step_id")
                    raw = trace.get("raw")
                    resolved = trace.get("resolved")
                    
                    with extraction_expander:
                        st.markdown(f"**Step ID:** `{step_id}`")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("Raw LLM Parameters")
                            st.json(raw)
                        with col2:
                            st.caption("Canonically Resolved Parameters")
                            st.json(resolved)
                        st.divider()
                        
                elif t_type == "tool_output":
                    has_graph_data = True
                    step_id = trace.get("step_id")
                    tool_name = trace.get("tool")
                    output = trace.get("output")
                    
                    with graph_expander:
                        st.markdown(f"**Step ID:** `{step_id}` | **Tool:** `{tool_name}`")
                        if isinstance(output, list) and len(output) > 5:
                            st.warning(f"Returned {len(output)} records. Showing first 5.")
                            st.json(output[:5])
                        else:
                            st.json(output)
                        st.divider()
                        
                elif t_type == "final_response":
                    status_text.empty()
                    with final_container:
                        st.success("✅ Execution Complete")
                        st.markdown("### Final Answer")
                        st.markdown(trace.get("data"))
                        
                elif t_type == "error":
                    st.error(f"Error at step {trace.get('step_id')}: {trace.get('error')}")
                    
            # Auto-collapse empty sections
            if not has_extraction:
                with extraction_expander:
                    st.info("No semantic extraction required for this query.")
            if not has_graph_data:
                with graph_expander:
                    st.info("No tools were called or no graph data was returned.")
                    
        except Exception as e:
            status_text.empty()
            st.error(f"An unexpected system error occurred: {str(e)}")
