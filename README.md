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

## 📂 Project Structure


## ⚡ How to Run Locally
    pip install -r requirements-dev.txt