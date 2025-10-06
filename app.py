import os
import glob
import streamlit as st
from pathlib import Path
from functools import lru_cache
import time
from typing import List

# LangChain + Vector store
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---- Config ----from pathlib import Path
import streamlit as st

# Resolve likely base directories so it works whether app.py is at repo root or in /app
APP_DIR = Path(__file__).resolve().parent
CWD = Path.cwd()

def find_base_dir():
    for base in [APP_DIR, APP_DIR.parent, CWD]:
        if (base / "data" / "raw").exists():
            return base
    return APP_DIR  # fallback

BASE_DIR = find_base_dir()
DATA_DIR = BASE_DIR / "data" / "raw"
INDEX_DIR = BASE_DIR / "data" / "index"

st.sidebar.write(f"🔎 Using data dir: `{DATA_DIR}`")
st.set_page_config(page_title="Architectural RAG Assistant", page_icon="🏗️", layout="wide")
st.title("🏗️ Architectural RAG Assistant")
st.caption("Prototype: Generative AI + Retrieval for sustainable design knowledge")

# Secrets / env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# ---- Helpers ----
def load_and_chunk_pdfs(pdf_paths):
    docs = []
    for p in pdf_paths:
        try:
            loader = PyPDFLoader(str(p))
            # OPTIONAL: limit long PDFs to first N pages to avoid huge indexes
            # pages = loader.load_and_split()[:25]  # or leave default
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
        except Exception as e:
            st.warning(f"Failed to parse {p.name}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,   # larger chunks => fewer chunks
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Hard cap to avoid rate limits on first build
    MAX_CHUNKS = 120
    if len(chunks) > MAX_CHUNKS:
        st.warning(f"Corpus is large ({len(chunks)} chunks). Indexing first {MAX_CHUNKS} chunks to avoid rate limits.")
        chunks = chunks[:MAX_CHUNKS]

    for c in chunks:
        c.metadata.setdefault("source", c.metadata.get("source", ""))
        c.metadata.setdefault("page", c.metadata.get("page", -1))
    return chunks

def make_embedder():
    # smaller, cheaper, higher limits
    return OpenAIEmbeddings(model="text-embedding-3-small", max_retries=8, timeout=60)

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def rate_limited_embed(texts: List[str], embedder: OpenAIEmbeddings, batch_size: int = 32):
    """
    Embed texts in small batches with exponential backoff to avoid RateLimitError.
    """
    vectors = []
    pause = 1.0  # start with 1s, will grow on 429
    for batch in batched(texts, batch_size):
        while True:
            try:
                vecs = embedder.embed_documents(batch)
                vectors.extend(vecs)
                # be polite: brief pause between batches
                time.sleep(0.25)
                break
            except Exception as e:
                msg = str(e).lower()
                if "rate" in msg or "429" in msg or "limit" in msg:
                    time.sleep(pause)
                    pause = min(pause * 1.8, 12.0)  # cap backoff
                else:
                    # surface unexpected errors
                    raise
    return vectors

@st.cache_resource(show_spinner=False)
def build_chroma_index(chunks, index_dir: str):
    from langchain_community.vectorstores import Chroma

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"doc-{i}" for i in range(len(chunks))]

    embedder = make_embedder()
    embs = rate_limited_embed(texts, embedder, batch_size=24)

    # Create (or open) the store *without* an embedding function (we provide embeddings)
    vs = Chroma(collection_name="docs", persist_directory=index_dir)
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids, embeddings=embs)
    vs.persist()
    return True



def ensure_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # If index already exists, skip embedding entirely
    if any(INDEX_DIR.iterdir()):
        return
    pdfs = list(DATA_DIR.glob("*.pdf"))
    st.sidebar.write(f"📚 Found PDFs: {[p.name for p in pdfs]}")
    if not pdfs:
        st.info("➕ No PDFs found. Add 3–5 public PDFs to `data/raw/` and click **Rebuild Index**.")
        return

    chunks = load_and_chunk_pdfs(pdfs)
    if not chunks:
        st.warning("Parsed 0 chunks. Check your PDFs (avoid scanned/image-only docs).")
        return

    with st.spinner(f"Building vector index… ({len(chunks)} chunks)"):
        build_chroma_index(chunks, str(INDEX_DIR))


def rebuild_index():
    # Clear and rebuild
    for item in INDEX_DIR.glob("*"):
        if item.is_file():
            item.unlink()
        else:
            import shutil
            shutil.rmtree(item)
    ensure_index()

def get_retriever():
    if not INDEX_DIR.exists() or not any(INDEX_DIR.iterdir()):
        st.warning("Index not built yet. Add PDFs to `data/raw/` and click **Rebuild Index**.")
        return None
    embed = make_embedder()
    vs = Chroma(persist_directory=str(INDEX_DIR), embedding_function=embedder)
    return vs.as_retriever(search_kwargs={"k": 5})


# ---- Sidebar ----
st.sidebar.header("Dataset & Index")
st.sidebar.write(f"Data folder: `{DATA_DIR}`")
if st.sidebar.button("🔁 Rebuild Index"):
    rebuild_index()
    st.sidebar.success("Index rebuilt.")

ensure_index()

with st.sidebar.expander("Example queries"):
    st.write("- Precedents for low-embodied carbon retrofit")
    st.write("- Daylight strategies with pros/cons")
    st.write("- Facade materials meeting fire-safety guidance")

# ---- Main QA UI ----
user_q = st.text_input("Ask a question about sustainable design / architecture docs:", "")
colA, colB = st.columns([2,1])

with colB:
    st.subheader("Settings")
    temperature = st.slider("Creativity", 0.0, 1.0, 0.0, 0.1)
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])

# Guard: require key for LLM
if not OPENAI_API_KEY:
    st.warning("Set your `OPENAI_API_KEY` in Streamlit Secrets or environment to enable answers. Retrieval still works (sources shown).")

# Build chain lazily
retriever = get_retriever()
if retriever is None:
    st.stop()

prompt_tpl = """
You are an assistant for architects. Answer ONLY from the provided context.
Cite sources inline like (Title p.X) or (Filename p.X). If the context is insufficient, say so and suggest a better query.

Question: {question}

Context:
{context}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_tpl)

def run_rag(query: str):
    # Build a QA chain that returns sources
    llm = ChatOpenAI(model=model_name, temperature=temperature, openai_api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    # Manual compose: retrieve -> stuff -> call model
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "I couldn't find relevant context in the current dataset. Try another query or add more PDFs.", []

    context_blocks = []
    for d in docs:
        src = d.metadata.get("source", "")
        page = d.metadata.get("page", "")
        label = f"{Path(src).name if src else 'doc'} p.{page}"
        context_blocks.append(f"[{label}] {d.page_content[:1200]}")

    filled = prompt.format(question=query, context="\n\n".join(context_blocks))
    if llm:
        resp = llm.invoke(filled)
        answer = resp.content
    else:
        # No LLM: show top snippets as a fallback
        answer = "LLM is disabled (no API key). Showing top retrieved snippets from your documents:\n\n" + "\n\n---\n\n".join(context_blocks[:3])

    return answer, docs

if st.button("Ask") or user_q.strip():
    q = user_q.strip() or "Precedents for low-embodied carbon retrofit"
    with st.spinner("Thinking…"):
        answer, docs = run_rag(q)
    with colA:
        st.subheader("Answer")
        st.write(answer)
        st.divider()
        st.subheader("Citations / Sources")
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "")
            page = d.metadata.get("page", "")
            st.markdown(f"**{i}.** `{Path(src).name}` — page **{page}**")
        st.divider()
        with st.expander("Show retrieved context"):
            for i, d in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}** — `{Path(d.metadata.get('source','')).name}` p.{d.metadata.get('page','')}")
                st.write(d.page_content[:1500])
                st.write("---")
else:
    st.info("Type a question above and press **Ask**. Add PDFs to `data/raw/` for better results.")






