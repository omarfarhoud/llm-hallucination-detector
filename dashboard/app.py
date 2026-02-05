"""
LLM Hallucination Detector - Streamlit Dashboard

This dashboard provides a graphical user interface (GUI) to interact with the 
Hallucination Detection API. It allows users to input questions, source contexts, 
and LLM-generated answers to evaluate faithfulness.

The dashboard visualizes:
1. The aggregated hallucination score.
2. Individual signal breakdowns (LLM-as-a-Judge, Semantic Similarity, Citation Check).
3. Detailed reasoning provided by the LLM Judge.

Architecture:
- Frontend: Streamlit (running on port 8501)
- Backend: FastAPI (running on port 8000)
- LLM Provider: Ollama (phi3:mini)
"""

import streamlit as st
import requests
import pandas as pd
import json

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="LLM Hallucination Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# Sidebar Settings
# ============================================================================
st.sidebar.header("⚙️ Configuration")
backend_url = st.sidebar.text_input("Backend API URL", "http://api:8000/detect")
threshold = st.sidebar.slider("Global Threshold", 0.0, 1.0, 0.7, help="Scores above this are flagged as hallucinations.")

st.sidebar.divider()
st.sidebar.info(
    "**Weights used by Backend:**\n"
    "- 🧠 **LLM Judge**: 60%\n"
    "- 🤖 **Similarity**: 25%\n"
    "- 📄 **Citations**: 15%"
)

# ============================================================================
# Main UI
# ============================================================================
st.title("🔍 LLM Hallucination Detector")
st.markdown("---")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 Input Data")
    
    question = st.text_input(
        "Question", 
        "What are the risk factors for type 2 diabetes?"
    )
    
    answer = st.text_area(
        "Generated Answer", 
        "Risk factors include obesity, lack of exercise, and genetics [2].",
        height=100
    )
    
    context_str = st.text_area(
        "Context Documents (JSON List)", 
        '[{"id": 1, "text": "Type 2 diabetes is insulin resistance."}, {"id": 2, "text": "Risk factors include obesity, physical inactivity, and family history."}]',
        height=150
    )

with col_right:
    st.subheader("📊 Analysis Results")
    
    if st.button("Analyze Faithfulness", use_container_width=True, type="primary"):
        try:
            # Parse Context
            context_data = json.loads(context_str)
            
            payload = {
                "question": question,
                "answer": answer,
                "context": context_data
            }
            
            with st.spinner("Processing parallel signals..."):
                response = requests.post(backend_url, json=payload)
                response.raise_for_status()
                res = response.json()

            # 1. Main Metrics
            score = res['hallucination_score']
            is_hallucinated = res['is_hallucination']
            verdict = "⚠️ HALLUCINATION" if is_hallucinated else "✅ FAITHFUL"
            verdict_color = "normal" if not is_hallucinated else "inverse"

            m1, m2 = st.columns(2)
            m1.metric("Final Score", f"{score:.3f}", delta=None)
            m2.metric("Verdict", verdict, delta=None, delta_color=verdict_color)

            # 2. Signal Breakdown Chart
            st.divider()
            st.markdown("#### Signal Hallucination Scores")
            signal_data = {
                "Signal": ["Semantic Similarity", "LLM Judge", "Citation Check"],
                "Score": [
                    res['signal_scores']['semantic_similarity_h'],
                    res['signal_scores']['llm_judge_h'],
                    res['signal_scores']['citation_check_h']
                ]
            }
            st.bar_chart(pd.DataFrame(signal_data).set_index("Signal"))

            # 3. Judge Reasoning (The highlight)
            st.markdown("#### 🧠 Judge Reasoning")
            reasoning = res['signals']['llm_judge']['raw']['details']
            st.success(reasoning) if not is_hallucinated else st.warning(reasoning)

        except json.JSONDecodeError:
            st.error("Invalid JSON in Context Documents. Please check your format.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
    else:
        st.write("Click 'Analyze Faithfulness' to begin.")

# ============================================================================
# Footer
# ============================================================================
st.divider()