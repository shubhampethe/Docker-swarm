import os
import logging
from typing import List, Tuple, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag.config import settings
from rag.embeddings import CustomContainerEmbeddings

logger = logging.getLogger(__name__)


class ChromaStoreManager:
    """
    Manages local persistent Chroma vector database and exposes
    LangChain Chroma vectorstore operations.
    """

    def __init__(
        self,
        embeddings: Optional[CustomContainerEmbeddings] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        self.embeddings = embeddings or CustomContainerEmbeddings()
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.chroma_collection_name
        self._vectorstore: Optional[Chroma] = None
        self._client: Optional[chromadb.ClientAPI] = None

        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

    def get_client(self) -> chromadb.ClientAPI:
        """
        Initializes and returns the local Chroma PersistentClient.
        """
        if self._client is None:
            logger.info(f"Initializing local Chroma PersistentClient at: {self.persist_directory}")
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
        return self._client

    def get_vectorstore(self) -> Chroma:
        """
        Returns an initialized LangChain Chroma vectorstore instance.
        """
        if self._vectorstore is not None:
            return self._vectorstore

        client = self.get_client()
        self._vectorstore = Chroma(
            client=client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Embeds and indexes LangChain Document chunks into local Chroma storage.
        """
        if not documents:
            return []

        vectorstore = self.get_vectorstore()
        ids = vectorstore.add_documents(documents)
        logger.info(
            f"Indexed {len(ids)} document chunks into local Chroma collection "
            f"'{self.collection_name}' at '{self.persist_directory}'."
        )
        return ids

    def search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Performs semantic similarity search with score.
        """
        vectorstore = self.get_vectorstore()
        return vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def get_retriever(self, k: int = 4):
        """
        Returns a LangChain retriever for LCEL chains.
        """
        vectorstore = self.get_vectorstore()
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def count(self) -> int:
        """
        Returns count of items currently in the local Chroma collection.
        """
        try:
            client = self.get_client()
            collection = client.get_collection(self.collection_name)
            return collection.count()
        except Exception:
            return 0

    def ping(self) -> bool:
        """
        Health check verifying local Chroma client initialization.
        """
        try:
            client = self.get_client()
            client.heartbeat()
            return True
        except Exception:
            return False
