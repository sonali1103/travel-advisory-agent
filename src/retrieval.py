from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from databricks_langchain import DatabricksEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


_CONTENTS_RE = re.compile(r"######\s+CONTENTS.*?(?=\n##\s)", flags=re.DOTALL | re.IGNORECASE)
_ANCHOR_RE = re.compile(r"</?a[^>]*>", flags=re.IGNORECASE)


def _clean_markdown(text: str) -> str:
    """Remove obvious noise that harms retrieval quality."""
    text = _CONTENTS_RE.sub("", text)
    text = _ANCHOR_RE.sub("", text)
    return text.strip()


def _compact_header_meta(headers: dict) -> dict:
    """Turn header metadata into a small 'path'"""
    h1 = headers.get("Header 1")
    h2 = headers.get("Header 2")
    h3 = headers.get("Header 3")
    parts = [p for p in (h1, h2, h3) if p]
    meta = {}
    if h1:
        meta["section"] = h1
    if h2:
        meta["subsection"] = h2
    if h3:
        meta["subsubsection"] = h3
    if parts:
        meta["path"] = " > ".join(parts)
    return meta


@dataclass
class PolicyRetriever:
    """
    - builds an in-memory FAISS index ONCE per notebook session
    - then supports search() for tool calls
    """
    policy_text: str
    k: int = 3
    embedding_endpoint: str = "databricks-bge-large-en"

    chunk_size: int = 1300
    chunk_overlap: int = 200

    _vs: FAISS | None = None

    def __post_init__(self) -> None:
        self.policy_text = _clean_markdown(self.policy_text)
        self._vs = self._build_index(self.policy_text)

    def _embedder(self) -> DatabricksEmbeddings:
        return DatabricksEmbeddings(endpoint=self.embedding_endpoint)

    def _build_index(self, markdown_text: str) -> FAISS:
        # 1) Split by markdown headers first (keeps structure)
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("##", "Header 1"),
                ("###", "Header 2"),
                ("######", "Header 3"),
            ]
        )
        header_docs = header_splitter.split_text(markdown_text)

        # 2) Chunk within each header section
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators = ["\n## ", "\n### ", "\n###### ", "\n**", " "]
        )

        docs: List[Document] = []
        for base_doc in header_docs:
            base_meta = _compact_header_meta(base_doc.metadata)
            chunks = chunker.split_text(base_doc.page_content)
            for i, chunk in enumerate(chunks):
                meta = dict(base_meta)
                meta["chunk"] = i
                docs.append(Document(page_content=chunk, metadata=meta))

        return FAISS.from_documents(docs, self._embedder())

    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        if not self._vs:
            raise RuntimeError("PolicyRetriever index not initialized.")
        return self._vs.similarity_search(query, k=k or self.k)

    def search_with_scores(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        if not self._vs:
            raise RuntimeError("PolicyRetriever index not initialized.")
        return self._vs.similarity_search_with_score(query, k=k or self.k)
