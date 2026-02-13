"""Enterprise Multimodal RAG (Text + Vision).

Run:
    streamlit run app.py
"""
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import streamlit as st

from config import AppConfig
from rag.chunking import extract_text_from_file
from rag.embeddings import JinaV4Embedder
from rag.llm import GroqAnswerGenerator
from rag.retriever import Chunk, MultiModalRetriever
from rag.reranker import SimpleReranker
from rag.vision import GroqVisionCaptioner


st.set_page_config(page_title="Multimodal RAG Studio", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
:root {
  --bg-a: #fdf6e3;
  --bg-b: #eaf4ff;
  --ink: #111827;
  --card: rgba(255,255,255,0.82);
  --brand: #0f766e;
  --brand2: #0ea5a3;
}
.stApp {
  background: radial-gradient(circle at 10% 10%, rgba(14,165,163,0.15), transparent 30%),
              radial-gradient(circle at 80% 15%, rgba(14,116,144,0.13), transparent 30%),
              linear-gradient(120deg, var(--bg-a), var(--bg-b));
  color: var(--ink);
}
.hero {
  border: 1px solid rgba(17,24,39,.1);
  border-radius: 18px;
  padding: 16px;
  background: var(--card);
  box-shadow: 0 10px 30px rgba(17,24,39,.08);
  animation: in .4s ease;
}
.card {
  border: 1px solid rgba(17,24,39,.08);
  border-radius: 12px;
  background: var(--card);
  padding: 10px;
}
.stButton>button {
  border: none;
  border-radius: 10px;
  font-weight: 700;
  color: #fff;
  background: linear-gradient(135deg, var(--brand), var(--brand2));
}
@keyframes in {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {}

st.markdown(
    """
<div class="hero">
  <h1 style="margin:0;">Enterprise Multimodal RAG</h1>
  <p style="margin:.4rem 0 0 0;">Grounded QA over text documents and images with Groq + Jina + FAISS.</p>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1.7], gap="large")

with left:
    st.subheader("Setup")
    env_cfg = AppConfig.from_env()

    groq_key = st.text_input("GROQ_API_KEY", type="password", value=env_cfg.groq_api_key)
    jina_key = st.text_input("JINA_API_KEY", type="password", value=env_cfg.jina_api_key)

    text_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])
    image_file = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg", "webp"])

    c1, c2 = st.columns(2)
    with c1:
        chunk_size = st.slider("Chunk size", min_value=120, max_value=600, value=300, step=20)
    with c2:
        overlap = st.slider("Chunk overlap", min_value=20, max_value=200, value=60, step=10)

    build_btn = st.button("Build Knowledge Base", use_container_width=True)

    if build_btn:
        if not groq_key or not jina_key:
            st.error("Both GROQ_API_KEY and JINA_API_KEY are required.")
        elif not text_file and not image_file:
            st.error("Upload at least one file (text or image).")
        else:
            build_start = time.perf_counter()
            embedder = JinaV4Embedder(api_key=jina_key)
            retriever = MultiModalRetriever(embedder=embedder, chunk_size=chunk_size, chunk_overlap=overlap)

            try:
                if text_file:
                    suffix = Path(text_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
                        t.write(text_file.getbuffer())
                        tpath = t.name
                    text = extract_text_from_file(tpath)
                    retriever.add_text(text, source=text_file.name)

                if image_file:
                    vision = GroqVisionCaptioner(api_key=groq_key, model=env_cfg.vision_model)
                    caption = vision.caption_bytes(image_file.getvalue(), mime=image_file.type or "image/png")
                    retriever.add_image_caption(caption, source=image_file.name)

                if retriever.total_chunks == 0:
                    st.error("No chunks were created. Check your files.")
                else:
                    retriever.build()
                    st.session_state.retriever = retriever
                    st.session_state.chat = []
                    st.session_state.metrics = {
                        "build_latency_sec": round(time.perf_counter() - build_start, 3),
                        "chunks": retriever.total_chunks,
                    }
                    st.success(f"Index ready with {retriever.total_chunks} chunks.")
            except Exception as exc:
                st.error(f"Failed to build knowledge base: {exc}")

    if st.session_state.retriever:
        m = st.session_state.metrics
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"Chunks: **{m.get('chunks', 0)}**")
        st.write(f"Build latency: **{m.get('build_latency_sec', 0)} sec**")
        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.subheader("Chat")
    modality = st.selectbox("Metadata filter", options=["both", "text", "image"], index=0)
    top_k = st.slider("Top-k", min_value=1, max_value=8, value=4)

    for msg in st.session_state.chat[-12:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("contexts"):
                with st.expander("Retrieved context"):
                    for i, c in enumerate(msg["contexts"], 1):
                        st.markdown(f"**{i}. [{c.modality}] {c.source}**\n\n{c.text[:500]}...")

    q = st.chat_input("Ask a question about your text/image context...")
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        retriever = st.session_state.retriever
        if not retriever:
            msg = "Build the knowledge base first."
            st.session_state.chat.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
        else:
            gen_start = time.perf_counter()
            llm = GroqAnswerGenerator(api_key=groq_key, model=env_cfg.llm_model)
            reranker = SimpleReranker()

            try:
                retrieved = retriever.search(q, top_k=top_k, modality_filter=modality)
                reranked = reranker.rank(q, retrieved)[:top_k]
                answer = llm.answer(query=q, contexts=reranked, max_history=4, history=st.session_state.chat)
                latency = round(time.perf_counter() - gen_start, 3)

                st.session_state.chat.append(
                    {
                        "role": "assistant",
                        "content": f"{answer}\n\n`Latency: {latency} sec`",
                        "contexts": reranked,
                    }
                )
                with st.chat_message("assistant"):
                    st.markdown(f"{answer}\n\n`Latency: {latency} sec`")
                    with st.expander("Retrieved context"):
                        for i, c in enumerate(reranked, 1):
                            st.markdown(f"**{i}. [{c.modality}] {c.source}**\n\n{c.text[:500]}...")
            except Exception as exc:
                err = f"Error: {exc}"
                st.session_state.chat.append({"role": "assistant", "content": err})
                with st.chat_message("assistant"):
                    st.error(err)
