"""
RAG-Based Chatbot Backend
FastAPI + LangChain + FAISS + Groq
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv
# Load .env relative to the location of rag.py
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")   # set in env or replace here
UPLOAD_DIR   = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
# Each session holds its own vector store, chain, and chat history.
# Structure: { session_id: { "chain": ..., "history": [...] } }

sessions: dict = {}

# ── Shared embedding model (loaded once) ─────────────────────────────────────

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_document(file_path: Path):
    """Load a document and return LangChain Document objects."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif ext == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif ext == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


SYSTEM_PROMPT_TEMPLATE = """You are a helpful document assistant. Answer the user's question using the context extracted from their uploaded documents.

Context from documents:
{context}

Instructions:
- Read the context carefully and answer the question in a clean, well-structured way.
- Synthesize and summarize the relevant information — do NOT copy raw text, tables, figure captions, or table-of-contents entries from the context.
- Write your answer in plain, readable prose or bullet points. Keep it concise and to the point.
- If the context only partially covers the question, answer with what is available.
- Only respond with "That question falls outside the scope of the uploaded documents. I can only help with topics covered in the files you've provided." if the context has absolutely zero relevance to the question.
- Never refuse to answer if the context contains ANY mention of the topic.
- Do not invent facts not present in the context."""

def build_chain(vectorstore: FAISS, memory: ConversationBufferMemory):
    """Build a ConversationalRetrievalChain from a vector store."""
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
    )

    # Custom prompt that enforces document-scoped answers
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context"],
                template=SYSTEM_PROMPT_TEMPLATE,
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["question"],
                template="{question}",
            )
        ),
    ])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="mmr",          # Max Marginal Relevance — diverse + relevant chunks
            search_kwargs={"k": 6, "fetch_k": 20},
        ),
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )
    return chain

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str
    history: List[dict]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload one or more documents.
    Returns a session_id that must be used for subsequent /chat calls.
    """
    # Validate extensions
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
            )

    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    all_docs = []
    saved_files = []

    for upload in files:
        dest = session_dir / upload.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        saved_files.append(upload.filename)

        try:
            docs = load_document(dest)
            all_docs.extend(docs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading {upload.filename}: {e}")

    if not all_docs:
        raise HTTPException(status_code=400, detail="No text could be extracted from the uploaded files.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Filter out noisy chunks (TOC entries, figure captions, pure number tables)
    def is_meaningful_chunk(doc) -> bool:
        text = doc.page_content.strip()
        if len(text) < 80:                          # too short to be useful
            return False
        words = text.split()
        if len(words) == 0:
            return False
        # Reject chunks where more than 40% of tokens are numbers/dots/special chars
        noise_tokens = sum(1 for w in words if not any(c.isalpha() for c in w))
        if noise_tokens / len(words) > 0.4:
            return False
        return True

    chunks = [c for c in chunks if is_meaningful_chunk(c)]

    # Build vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Build memory + chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = build_chain(vectorstore, memory)

    sessions[session_id] = {
        "chain": chain,
        "memory": memory,
        "history": [],
        "files": saved_files,
    }

    return {
        "session_id": session_id,
        "files": saved_files,
        "chunks": len(chunks),
        "message": "Files processed successfully. You can now ask questions.",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Ask a question against the uploaded documents.
    Maintains conversation history per session.
    """
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload files first.")

    session = sessions[req.session_id]
    chain   = session["chain"]

    try:
        result = chain.invoke({"question": req.question})
        answer = result.get("answer", "Sorry, I could not generate an answer.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Store history
    session["history"].append({"role": "user",      "content": req.question})
    session["history"].append({"role": "assistant",  "content": answer})

    return ChatResponse(answer=answer, history=session["history"])


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    """Return the full chat history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"history": sessions[session_id]["history"]}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up a session and its uploaded files."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sessions.pop(session_id, None)
    session_dir = UPLOAD_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    return {"message": "Session deleted."}


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Serve frontend ────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag:app", host="127.0.0.1", port=8000, reload=True)
