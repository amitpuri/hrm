# hier_rag_langchain.py
# ---------------------------------------------
# Hierarchical RAG in LangChain (two-stage retrieval: sections -> fine chunks)
# - Uses langchain-chroma (no deprecation warnings)
# - Global (summary) index narrows search to candidate sections
# - Fine-grained index retrieves precise spans from those sections
# - LLM compression + embedding de-dup via DocumentCompressorPipeline
# - Clean LCEL answer chain with lightweight citations
# ---------------------------------------------

import os
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# Prefer new package; gracefully fall back if not installed (will warn)
try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover
    from langchain_community.vectorstores import Chroma  # type: ignore

from langchain.retrievers.document_compressors import (
    LLMChainExtractor,
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)

# -------------------------
# Provider helpers
# -------------------------
def get_llm():
    provider = os.getenv("PROVIDER", "openai").strip().lower()
    if provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.2,
            streaming=True,
            timeout=60,
            max_retries=2,
        )
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        streaming=True,
        timeout=60,
        max_retries=2,
    )


def get_embeddings():
    provider = os.getenv("PROVIDER", "openai").strip().lower()
    if provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"))

# -------------------------
# Loading & structuring docs
# -------------------------
def load_docs(data_dir: str) -> List[Document]:
    loaders = [
        DirectoryLoader(data_dir, glob="**/*.md"),
        DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}),
        DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]
    docs: List[Document] = []
    for ld in loaders:
        try:
            docs.extend(ld.load())
        except Exception as e:
            print(f"[warn] loader {ld} error: {e}")
    return docs


def split_into_sections(docs: List[Document]) -> List[Document]:
    section_docs: List[Document] = []
    for d in docs:
        source = d.metadata.get("source", "")
        if source.endswith(".md"):
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")])
            parts = splitter.split_text(d.page_content)
            for p in parts:
                meta = dict(d.metadata)
                meta.update(p.metadata)
                section_docs.append(Document(page_content=p.page_content, metadata=meta))
        else:
            rcs = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
            section_docs.extend(rcs.split_documents([d]))
    for sd in section_docs:
        sd.metadata["section_id"] = sd.metadata.get("section_id", str(uuid.uuid4()))
        sd.metadata["source"] = sd.metadata.get("source", "unknown")
    return section_docs


def split_into_fine_chunks(section_docs: List[Document]) -> List[Document]:
    fine_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "])
    fine_chunks: List[Document] = []
    for sec in section_docs:
        base_title = sec.metadata.get("h1") or sec.metadata.get("title") or os.path.basename(sec.metadata.get("source", ""))
        for ch in fine_splitter.split_text(sec.page_content):
            fine_chunks.append(
                Document(
                    page_content=ch,
                    metadata={
                        "section_id": sec.metadata["section_id"],
                        "source": sec.metadata.get("source", "unknown"),
                        "title": base_title,
                    },
                )
            )
    return fine_chunks

# -------------------------
# Optional global summaries
# -------------------------
def make_section_summaries(section_docs: List[Document], llm) -> List[Document]:
    tmpl = ChatPromptTemplate.from_messages(
        [
            ("system", "Write a crisp 1–2 sentence summary preserving key facts, names, and definitions."),
            ("user", "Summarize this section for retrieval:\n\n{content}"),
        ]
    )
    chain = tmpl | llm | StrOutputParser()
    summaries: List[Document] = []
    for sec in section_docs:
        summary = chain.invoke({"content": sec.page_content}) or ""
        summaries.append(
            Document(
                page_content=summary.strip(),
                metadata={
                    "parent_id": sec.metadata["section_id"],
                    "source": sec.metadata.get("source", "unknown"),
                    "title": sec.metadata.get("h1")
                    or sec.metadata.get("title")
                    or os.path.basename(sec.metadata.get("source", "")),
                },
            )
        )
    return summaries

# -------------------------
# Parent section persistence (JSONL)
# -------------------------
def save_parents(sections: List[Document], persist_dir: str):
    out = Path(persist_dir) / "parents.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for sec in sections:
            rec = {"id": sec.metadata["section_id"], "metadata": sec.metadata, "content": sec.page_content}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_parents(persist_dir: str) -> Dict[str, Document]:
    path = Path(persist_dir) / "parents.jsonl"
    parent_map: Dict[str, Document] = {}
    if not path.exists():
        return parent_map
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                parent_map[rec["id"]] = Document(page_content=rec["content"], metadata=rec["metadata"])
            except Exception:
                continue
    return parent_map

# -------------------------
# Index build
# -------------------------
def build_indices(data_dir: str, persist_dir: str = "./chroma", use_summaries: bool = True):
    llm = get_llm()
    embed = get_embeddings()

    raw_docs = load_docs(data_dir)
    print(f"[info] loaded {len(raw_docs)} raw docs")

    sections = split_into_sections(raw_docs)
    print(f"[info] split into {len(sections)} sections")

    fine_chunks = split_into_fine_chunks(sections)
    print(f"[info] split into {len(fine_chunks)} fine chunks")

    fine_vs = Chroma(collection_name="fine_chunks", embedding_function=embed, persist_directory=persist_dir)
    fine_vs.add_documents(fine_chunks)

    if use_summaries:
        summaries = make_section_summaries(sections, llm)
    else:
        summaries = [
            Document(
                page_content=sec.page_content[:400],
                metadata={
                    "parent_id": sec.metadata["section_id"],
                    "source": sec.metadata.get("source", "unknown"),
                    "title": sec.metadata.get("h1")
                    or sec.metadata.get("title")
                    or os.path.basename(sec.metadata.get("source", "")),
                },
            )
            for sec in sections
        ]

    summary_vs = Chroma(collection_name="global_summaries", embedding_function=embed, persist_directory=persist_dir)
    summary_vs.add_documents(summaries)

    save_parents(sections, persist_dir)
    print("[info] indices built (Chroma auto-persisted).")

# -------------------------
# Stage-1 + Stage-2 retrieval
# -------------------------
def stage1_candidate_sections(query: str, k_sections: int, persist_dir: str) -> List[str]:
    embed = get_embeddings()
    summary_vs = Chroma(collection_name="global_summaries", embedding_function=embed, persist_directory=persist_dir)
    hits = summary_vs.similarity_search(query, k=k_sections)
    return list({d.metadata.get("parent_id") for d in hits if d.metadata.get("parent_id")})


def fine_search(query: str, candidate_section_ids: List[str], k: int, persist_dir: str) -> List[Document]:
    embed = get_embeddings()
    fine_vs = Chroma(collection_name="fine_chunks", embedding_function=embed, persist_directory=persist_dir)
    if candidate_section_ids:
        docs = fine_vs.similarity_search(query, k=k, filter={"section_id": {"$in": candidate_section_ids}})
        if len(docs) < max(4, k // 2):
            extra = fine_vs.similarity_search(query, k=k)
            seen = set((d.page_content, d.metadata.get("section_id")) for d in docs)
            for d in extra:
                key = (d.page_content, d.metadata.get("section_id"))
                if key not in seen:
                    docs.append(d)
                    seen.add(key)
    else:
        docs = fine_vs.similarity_search(query, k=k)
    return docs

# -------------------------
# Compression / de-dup (fixed for LC 0.2)
# -------------------------
def make_compressor():
    llm = get_llm()
    embed = get_embeddings()
    extractor = LLMChainExtractor.from_llm(
        llm,
        prompt=ChatPromptTemplate.from_template(
            "From the context, extract only the minimal spans strictly needed to answer: {question}\n\nContext:\n{context}"
        ),
    )
    dedupe = EmbeddingsFilter(embeddings=embed, similarity_threshold=0.76)
    # Chain them: extractor -> embedding de-dup
    return DocumentCompressorPipeline(transformers=[extractor, dedupe])

# -------------------------
# Answer chain (LCEL)
# -------------------------
def make_answer_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful assistant. Use ONLY the provided context. "
                "If the answer isn't in the context, say you don't know. "
                "Cite sources as [n] using the source/title metadata.",
            ),
            (
                "user",
                "Question: {question}\n\n"
                "Context:\n{context}\n\n"
                "Answer with brief reasoning, then bullet-pointed citations.",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()

# -------------------------
# Orchestrate a single query
# -------------------------
def answer_query(query: str, k_sections: int = 6, k_fine: int = 10, persist_dir: str = "./chroma") -> str:
    candidate_ids = stage1_candidate_sections(query, k_sections=k_sections, persist_dir=persist_dir)
    fine_hits = fine_search(query, candidate_ids, k=k_fine, persist_dir=persist_dir)

    # Build a single joined context for the extractor prompt
    joined = "\n\n---\n\n".join([d.page_content for d in fine_hits]) if fine_hits else ""
    compressor = make_compressor()
    try:
        filtered = compressor.compress_documents(
            fine_hits,
            query={"question": query, "context": joined},  # LLMChainExtractor needs both
        )
    except TypeError:
        # Some LC versions expect query=str; fall back gracefully
        filtered = compressor.compress_documents(fine_hits, query=query)
    except Exception as e:
        print(f"[warn] compression failed ({e}); falling back to raw hits")
        filtered = fine_hits

    def fmt(doc: Document, n: int) -> str:
        title = doc.metadata.get("title") or os.path.basename(doc.metadata.get("source", ""))
        src = doc.metadata.get("source", "")
        sid = doc.metadata.get("section_id", "")
        return f"[{n}] ({title}) {src} §{sid}\n{doc.page_content.strip()}"

    numbered = [fmt(d, i + 1) for i, d in enumerate(filtered[:8])]
    context = "\n\n".join(numbered) if numbered else "NO CONTEXT"

    chain = make_answer_chain()
    return chain.invoke({"question": query, "context": context})

# -------------------------
# CLI
# -------------------------
def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=str, help="Directory with source documents to index")
    ap.add_argument("--ask", type=str, help="Query to run against the hierarchical index")
    ap.add_argument("--persist", type=str, default="./chroma")
    ap.add_argument("--no-summaries", action="store_true", help="Do not use LLM summaries for stage-1")
    ap.add_argument("--k_sections", type=int, default=6, help="How many sections to consider from stage-1")
    ap.add_argument("--k_fine", type=int, default=10, help="How many fine chunks to retrieve")
    args = ap.parse_args()

    if args.build:
        build_indices(args.build, persist_dir=args.persist, use_summaries=not args.no_summaries)
        print("[done] index built")

    if args.ask:
        print("Q:", args.ask)
        ans = answer_query(args.ask, k_sections=args.k_sections, k_fine=args.k_fine, persist_dir=args.persist)
        print("\n---\nA:\n", ans)

if __name__ == "__main__":
    main()
