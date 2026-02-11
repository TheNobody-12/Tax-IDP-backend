"""
Vector store implementations for semantic similarity search.
Used for hybrid context retrieval in medical expense extraction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os

# Optional dependencies
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

import numpy as np


@dataclass
class RetrievedContext:
    """A retrieved context item with similarity score"""
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add(self, chunk_id: str, content: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> None:
        """Add a document to the store"""
        pass
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievedContext]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def query(self, doc_id: str, query_text: str, top_k: int = 5) -> List[RetrievedContext]:
        """Search for similar documents within a specific document"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the store"""
        pass
    
    @abstractmethod
    def clear_document(self, doc_id: str) -> None:
        """Clear all items from a specific document"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store"""
        pass


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store using sentence-transformers.
    Good for small documents without persistent storage needs.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._documents: List[Dict] = []
        self._embeddings: List[np.ndarray] = []
        self._model = None
        self._model_name = model_name
    
    def _get_model(self):
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
            self._model = SentenceTransformer(self._model_name)
        return self._model
    
    def _compute_embedding(self, text: str) -> np.ndarray:
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True)
    
    def add(self, chunk_id: str, content: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> None:
        embedding = self._compute_embedding(content)
        meta = metadata or {}
        if doc_id:
            meta["doc_id"] = doc_id
        self._documents.append({
            "chunk_id": chunk_id,
            "content": content,
            "metadata": meta
        })
        self._embeddings.append(embedding)
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievedContext]:
        return self._search_impl(query, top_k, doc_id=None)
    
    def query(self, doc_id: str, query_text: str, top_k: int = 5) -> List[RetrievedContext]:
        return self._search_impl(query_text, top_k, doc_id=doc_id)

    def _search_impl(self, query: str, top_k: int, doc_id: Optional[str] = None) -> List[RetrievedContext]:
        if not self._documents:
            return []
        
        query_embedding = self._compute_embedding(query)
        
        # Compute cosine similarity
        scores = []
        for i, doc_embedding in enumerate(self._embeddings):
            # Filter by doc_id if specified
            if doc_id and self._documents[i].get("metadata", {}).get("doc_id") != doc_id:
                continue
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scores.append((i, float(similarity)))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            doc = self._documents[idx]
            results.append(RetrievedContext(
                content=doc["content"],
                metadata=doc["metadata"],
                score=score,
                chunk_id=doc["chunk_id"]
            ))
        
        return results
    
    def clear(self) -> None:
        self._documents = []
        self._embeddings = []
    
    def clear_document(self, doc_id: str) -> None:
        indices_to_remove = [i for i, doc in enumerate(self._documents)
                            if doc.get("metadata", {}).get("doc_id") == doc_id]
        for i in reversed(indices_to_remove):
            self._documents.pop(i)
            self._embeddings.pop(i)
    
    def count(self) -> int:
        return len(self._documents)


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based vector store for persistent or in-memory storage.
    Suitable for larger document sets.
    """
    
    def __init__(
        self,
        collection_name: str = "medical_expenses",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        if not HAS_CHROMADB:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
        
        self._embedding_model_name = embedding_model
        self._embedding_model = None
        
        # Create client
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _get_embedding_model(self):
        if self._embedding_model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers not installed")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model
    
    def _embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def add(self, chunk_id: str, content: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> None:
        embedding = self._embed([content])[0]
        meta = metadata or {}
        if doc_id:
            meta["doc_id"] = doc_id
        self._collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta]
        )
    
    def add_batch(self, items: List[Dict], doc_id: Optional[str] = None) -> None:
        """Add multiple items at once for efficiency"""
        if not items:
            return
        
        ids = [item["chunk_id"] for item in items]
        contents = [item["content"] for item in items]
        metadatas = []
        for item in items:
            meta = item.get("metadata", {})
            if doc_id:
                meta["doc_id"] = doc_id
            metadatas.append(meta)
        embeddings = self._embed(contents)
        
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievedContext]:
        return self._search_impl(query, top_k, doc_id=None)
    
    def query(self, doc_id: str, query_text: str, top_k: int = 5) -> List[RetrievedContext]:
        return self._search_impl(query_text, top_k, doc_id=doc_id)

    def _search_impl(self, query: str, top_k: int, doc_id: Optional[str] = None) -> List[RetrievedContext]:
        if self._collection.count() == 0:
            return []
        
        query_embedding = self._embed([query])[0]
        
        # Build where clause for doc_id filter
        where_clause = {"doc_id": doc_id} if doc_id else None
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where_clause
        )
        
        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distance, convert to similarity (1 - distance for cosine)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance
                
                retrieved.append(RetrievedContext(
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    score=similarity,
                    chunk_id=chunk_id
                ))
        
        return retrieved
    
    def clear(self) -> None:
        # Delete all items
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
    
    def clear_document(self, doc_id: str) -> None:
        # Delete items for a specific document
        results = self._collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
    
    def count(self) -> int:
        return self._collection.count()


class MockVectorStore(VectorStore):
    """
    Mock vector store that doesn't require any embeddings.
    Returns empty results - useful for testing or when embeddings are not needed.
    """
    
    def __init__(self):
        self._documents: Dict[str, Dict] = {}
    
    def add(self, chunk_id: str, content: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> None:
        meta = metadata or {}
        if doc_id:
            meta["doc_id"] = doc_id
        self._documents[chunk_id] = {
            "content": content,
            "metadata": meta
        }
    
    def search(self, query: str, top_k: int = 5) -> List[RetrievedContext]:
        return self._search_impl(query, top_k, doc_id=None)
    
    def query(self, doc_id: str, query_text: str, top_k: int = 5) -> List[RetrievedContext]:
        return self._search_impl(query_text, top_k, doc_id=doc_id)

    def _search_impl(self, query: str, top_k: int, doc_id: Optional[str] = None) -> List[RetrievedContext]:
        # Filter by doc_id if specified
        filtered = [(cid, doc) for cid, doc in self._documents.items()
                   if not doc_id or doc.get("metadata", {}).get("doc_id") == doc_id]
        
        results = []
        for i, (chunk_id, doc) in enumerate(filtered[:top_k]):
            results.append(RetrievedContext(
                content=doc["content"],
                metadata=doc["metadata"],
                score=1.0 - (i * 0.1),  # Decreasing fake scores
                chunk_id=chunk_id
            ))
        return results
    
    def clear(self) -> None:
        self._documents = {}
    
    def clear_document(self, doc_id: str) -> None:
        to_remove = [cid for cid, doc in self._documents.items()
                    if doc.get("metadata", {}).get("doc_id") == doc_id]
        for cid in to_remove:
            del self._documents[cid]
    
    def count(self) -> int:
        return len(self._documents)


def create_vector_store(
    store_type: str = "memory",
    **kwargs
) -> VectorStore:
    """Factory function to create a vector store"""
    if store_type == "memory":
        return InMemoryVectorStore(**kwargs)
    elif store_type == "chroma":
        return ChromaVectorStore(**kwargs)
    elif store_type == "mock":
        return MockVectorStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
