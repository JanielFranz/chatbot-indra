# 📚 Multimodal RAG Chatbot (FastAPI + Gradio + FAISS)

This project is the final challenge for building a **multimodal RAG chatbot** that answers questions about a PDF document (text + images).  
The chatbot uses **FastAPI** for the backend, **Gradio** for the frontend, and a **vector database (FAISS)** to store embeddings of the document.

---

## 🚀 Project Overview

1. **Preprocess Document**
   - Extract **text** and **images** from the PDF.
   - Chunk text into smaller sections.
   - Generate **embeddings** for each text chunk.
   - Save them into a **vector database (FAISS)** along with metadata linking images to text.

2. **Backend (FastAPI)**
   - Exposes an API for:
     - Querying the vector DB with a user’s question.
     - Retrieving the most relevant text chunks + images.
     - Calling an LLM (OpenAI / Hugging Face / Groq) to generate an answer.
   - Returns the answer + related image(s).

3. **Frontend (Gradio)**
   - Simple chat interface.
   - User asks a question → sends it to FastAPI → displays the text + image response.

4. **Deployment**
   - Deploy the FastAPI + Gradio app to **Render**, **Railway**, or **Hugging Face Spaces**.
   - Share the public URL.

---

## 🛠️ Tech Stack

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/)  
- **Frontend:** [Gradio](https://www.gradio.app/)  
- **Vector DB:** [FAISS](https://github.com/facebookresearch/faiss) (or [ChromaDB](https://www.trychroma.com/))  
- **LLM & Embeddings:**  
  - OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)  
  - OR Hugging Face (`all-MiniLM-L6-v2`, etc.)  
  - OR Groq API  

---

# 📂 Project Structure

```
src/
├── __init__.py
├── main.py                     # FastAPI app entry
├── container.py                 # Dependency injection config
├── logging.py  
|-- .env
├── app/                         # Core domain apps (grouped here)
│   ├── __init__.py
│   │
│   ├── ingestion/               # Document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   ├── service.py
│   │   ├── models.py
│   │   └── processors/
│   │       ├── __init__.py
│   │       ├── pdf_processor.py
│   │       └── image_processor.py
│   │
│   └── chatbot/                 # Chatbot logic
│       ├── __init__.py
│       ├── controller.py
│       ├── service.py
│       ├── models.py
│       └── rag_pipeline.py      # Full RAG pipeline
│
├── guardrails/
│   ├── __init__.py
│   ├── input/
│   │   ├── __init__.py
│   │   ├── base_guard.py
│   │   ├── content_filter.py
│   │   ├── length_validator.py
│   │   └── language_detector.py
│   ├── output/
│   │   ├── __init__.py
│   │   ├── hallucination_detector.py
│   │   └── safety_filter.py
│   └── pipeline.py              # Guardrails orchestration
│
├── llm/
│   ├── __init__.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── groq_provider.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── templates.py
│   │   └── prompt_manager.py
│   └── chain_manager.py         # LLM orchestration
│
├── ui/
│   ├── __init__.py
│   ├── gradio/
│   │   ├── __init__.py
│   │   ├── app.py                # Gradio entry
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── chat_interface.py
│   │   │   └── upload_interface.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── gradio_helpers.py
│   └── static/
│       ├── css/
│       ├── js/
│       └── images/
│
├── utils/
│   ├── __init__.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── batch_processor.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_processor.py
│   │   └── pdf_processor.py
│   └── helpers/
│       ├── __init__.py
│       ├── file_utils.py
│       └── text_utils.py
│
└── tests/
    ├── __init__.py
    ├── unit/
    ├── integration/
    └── fixtures/
```


## ⚡ How to Run Locally
    pip install -r requirements-dev.txt