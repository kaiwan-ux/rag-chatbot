# RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI**, **LangChain**, **FAISS**, and **Groq**.

Upload PDF, DOCX, or TXT files and ask questions about them. The app maintains conversation history so follow-up questions work naturally.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI |
| LLM | Groq (`llama3-8b-8192`) — free tier available |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace (runs locally, no API key needed) |
| Vector Store | FAISS (in-memory) |
| Orchestration | LangChain |
| Frontend | Plain HTML/CSS/JS (served by FastAPI) |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key

Get a free key at https://console.groq.com

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Or export it directly:

```bash
export GROQ_API_KEY=your_key_here   # Windows: set GROQ_API_KEY=your_key_here
```

### 5. Run the server

```bash
python rag.py
```

Open your browser at **http://localhost:8000**

---

## Usage

1. **Upload** one or more PDF / DOCX / TXT files using the upload panel.
2. Wait for the "processed successfully" message.
3. **Ask questions** in the chat box. Follow-up questions work because conversation history is maintained per session.
4. Click **New Session** to start fresh with different documents.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload files, returns `session_id` |
| `POST` | `/chat` | Ask a question (`session_id` + `question`) |
| `GET` | `/history/{session_id}` | Get full chat history |
| `DELETE` | `/session/{session_id}` | Delete session & uploaded files |
| `GET` | `/health` | Health check |

---

## Switching LLMs

The LLM is instantiated in `build_chain()` inside `rag.py`. To swap providers:

**OpenAI**
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
```

**Ollama (local)**
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
```

**Together AI**
```python
from langchain_together import ChatTogether
llm = ChatTogether(model="meta-llama/Llama-3-8b-chat-hf", api_key=os.getenv("TOGETHER_API_KEY"))
```

---

## Project Structure

```
rag-chatbot/
├── rag.py            # FastAPI backend + RAG pipeline
├── index.html        # Frontend UI
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── README.md
└── uploads/          # Created automatically for uploaded files
```
