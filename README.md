# Enterprise Multimodal RAG (Text + Vision)

End-to-end multimodal Retrieval-Augmented Generation system for grounded QA over:
- Text files (TXT, PDF)
- Images (diagrams, charts, screenshots)

## Features
- FAISS retrieval over Jina Embeddings v4
- Groq Vision image captioning
- Groq grounded answer generation
- Simple reranker
- Metadata filtering: `text`, `image`, `both`
- Session chat memory and latency tracking in Streamlit

## Project Structure

```bash
multimodal-rag-jina4/
├── app.py
├── requirements.txt
├── config.py
├── README.md
└── rag/
    ├── __init__.py
    ├── embeddings.py
    ├── retriever.py
    ├── chunking.py
    ├── vision.py
    ├── reranker.py
    └── llm.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set environment variables or provide them in the app UI:
- `GROQ_API_KEY`
- `JINA_API_KEY`

## Run

```bash
streamlit run app.py
```
