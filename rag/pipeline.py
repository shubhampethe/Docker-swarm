import os
import logging
import asyncio
from typing import Dict, Any, List, Optional
from rag.config import settings
from rag.embeddings import CustomContainerEmbeddings
from rag.llamaparse_loader import LlamaParseLoader
from rag.chroma_store import ChromaStoreManager
from rag.swarm_scaler import DockerSwarmScaler
from rag.chain import RAGChain

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main coordinator for the LangChain RAG Pipeline with LlamaIndex LlamaParse (asyncio),
    Custom Embedding Microservice with Docker Swarm Auto-Scaling, and Local Chroma Vector Store.
    """

    def __init__(
        self,
        llama_cloud_api_key: Optional[str] = None,
        embedding_api_url: Optional[str] = None,
        chroma_persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.embeddings = CustomContainerEmbeddings(api_url=embedding_api_url)
        self.loader = LlamaParseLoader(api_key=llama_cloud_api_key)
        self.store = ChromaStoreManager(
            embeddings=self.embeddings,
            persist_directory=chroma_persist_directory,
            collection_name=collection_name
        )
        self.scaler = DockerSwarmScaler()

    def health_check(self) -> Dict[str, Any]:
        """
        Verify status of LlamaParse, Embedding microservice, Swarm autoscaler, and local Chroma store.
        """
        # 1. Check LlamaParse Configuration
        llama_configured = self.loader.is_configured()

        # 2. Check Embedding Microservice Container
        embedding_status = "error"
        embedding_info = {}
        try:
            res = self.embeddings.ping()
            embedding_status = "ok"
            embedding_info = res
        except Exception as e:
            embedding_info = {"error": str(e)}

        # 3. Check Swarm Autoscaler
        swarm_active = self.scaler.is_swarm_active()
        current_replicas = self.scaler.get_current_replicas()

        # 4. Check Local Chroma Store
        chroma_ok = self.store.ping()
        doc_count = self.store.count()

        return {
            "llamaparse_parser": {
                "configured": llama_configured,
                "status": "ready (Async LlamaParse API)" if llama_configured else "fallback mode (local reader)",
                "result_type": self.loader.result_type
            },
            "embedding_container": {
                "url": self.embeddings.api_url,
                "reachable": embedding_status == "ok",
                "status": embedding_status,
                "details": embedding_info
            },
            "swarm_autoscaler": {
                "active": swarm_active,
                "service": self.scaler.service_name,
                "current_replicas": current_replicas,
                "scale_threshold": self.scaler.scale_threshold,
                "scale_up_target": self.scaler.scale_up_replicas,
                "status": f"Online ({current_replicas} replica(s))" if swarm_active and current_replicas is not None else "Swarm Inactive / Single Container"
            },
            "chroma_local_store": {
                "persist_directory": self.store.persist_directory,
                "ready": chroma_ok,
                "collection": self.store.collection_name,
                "total_documents": doc_count,
                "status": "ready" if chroma_ok else "error"
            }
        }

    async def aingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Asynchronous end-to-end ingestion:
        1. Async parse document via LlamaIndex LlamaParse (aload_data).
        2. Automatically scale Swarm service to 3 replicas if chunks >= threshold.
        3. Embed via custom embedding container across replicas and index into local Chroma.
        4. Automatically scale Swarm service back down to 1 replica.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        logger.info(f"Starting async ingestion for: {file_path}")

        # 1. Async parse via LlamaParse
        documents = await self.loader.aload_file(file_path)
        if not documents:
            return {
                "status": "warning",
                "message": "No extractable text content found in document.",
                "file": file_path,
                "chunks_indexed": 0
            }

        # 2. Embed & Store with automatic Swarm elasticity
        async with self.scaler.a_autoscale(len(documents)):
            ids = self.store.add_documents(documents)

        return {
            "status": "success",
            "file": os.path.basename(file_path),
            "file_path": file_path,
            "chunks_created": len(documents),
            "chunks_indexed": len(ids),
            "total_collection_count": self.store.count(),
            "replicas_used": list(self.embeddings.seen_replicas)
        }

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Synchronous end-to-end ingestion with automatic Swarm scaling (1 -> 3 -> 1).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        logger.info(f"Starting ingestion for: {file_path}")

        # 1. Parse via loader
        documents = self.loader.load_file(file_path)
        if not documents:
            return {
                "status": "warning",
                "message": "No extractable text content found in document.",
                "file": file_path,
                "chunks_indexed": 0
            }

        # 2. Embed & Store with automatic Swarm elasticity
        with self.scaler.autoscale(len(documents)):
            ids = self.store.add_documents(documents)

        return {
            "status": "success",
            "file": os.path.basename(file_path),
            "file_path": file_path,
            "chunks_created": len(documents),
            "chunks_indexed": len(ids),
            "total_collection_count": self.store.count(),
            "replicas_used": list(self.embeddings.seen_replicas)
        }

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Execute RAG retrieval and generate answer for the question using Groq / configured LLM.
        """
        retriever = self.store.get_retriever(k=top_k)
        chain = RAGChain(retriever=retriever)
        return chain.answer(question)

    def search_similar(self, query: str, top_k: int = 4):
        """
        Retrieve raw matching chunks and similarity scores without synthesis.
        """
        return self.store.search(query=query, k=top_k)
