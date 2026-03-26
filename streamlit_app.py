import streamlit as st

from evaluation.benchmark_dataset import benchmark
from evaluation.evaluator import RAGEvaluator
from evaluation.report import summarize
from orchestration.pipeline import build_pipeline


st.set_page_config(
    page_title="Elite-RAG Demo",
    page_icon=":robot_face:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_pipeline(quickstart: bool):
    return build_pipeline(quickstart=quickstart)


def reset_chat():
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("Elite-RAG Live Demo")
st.caption("Interactive RAG interface for quick demos and live walkthroughs.")

with st.sidebar:
    st.header("Demo Controls")
    quickstart = st.toggle(
        "Quickstart mode (recommended)",
        value=True,
        help="Uses lightweight CPU-safe defaults for reliable live demos.",
    )
    if st.button("Reset chat"):
        reset_chat()
        st.rerun()

    st.markdown("---")
    st.subheader("Evaluation")
    run_eval = st.button("Run benchmark evaluation")

pipeline = load_pipeline(quickstart=quickstart)

left, right = st.columns([2, 1])

with right:
    st.subheader("Mode")
    st.write("`Quickstart`" if quickstart else "`Full`")
    if quickstart:
        st.info("Portable mode for presentations and CPU-only environments.")
    else:
        st.warning("Full mode may require GPU/CUDA and larger model downloads.")

    if run_eval:
        with st.spinner("Running evaluation..."):
            evaluator = RAGEvaluator(pipeline, benchmark)
            results = evaluator.run()
            summary = summarize(results)
        st.success("Evaluation complete")
        st.json(summary)

with left:
    st.subheader("Ask a question")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask something about RAG...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline(user_prompt)
                answer = result["generation"]
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
