import time
import logging
from typing import List, Set, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.embeddings import Embeddings
from rag.config import settings

logger = logging.getLogger(__name__)


class CustomContainerEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation that sends requests to the containerized
    embedding microservice hosting sentence-transformers/all-MiniLM-L6-v2.
    Supports Docker Swarm scaling (replicas: 3) with load-balancing and replica tracking.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        batch_size: Optional[int] = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.api_url = (api_url or settings.embedding_api_url).rstrip("/")
        self.batch_size = batch_size or settings.embedding_batch_size
        self.timeout = timeout
        self.seen_replicas: Set[str] = set()

        # Resilient HTTP session with connection pooling and retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents in batches across scaled Docker Swarm replicas.
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        endpoint = f"{self.api_url}/embed"
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Embedding {len(texts)} chunks in {total_batches} batches (batch_size={self.batch_size}) "
            f"via Swarm ingress at {self.api_url}..."
        )

        for i in range(0, len(texts), self.batch_size):
            batch_num = (i // self.batch_size) + 1
            batch = texts[i : i + self.batch_size]

            try:
                response = requests.post(
                    endpoint,
                    json={"texts": batch, "normalize": True},
                    headers={"Connection": "close"},
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                replica_id = data.get("replica_id", "unknown")
                self.seen_replicas.add(replica_id)

                logger.info(
                    f"✓ Batch {batch_num}/{total_batches} ({len(batch)} chunks) "
                    f"processed by Swarm replica: [{replica_id}]"
                )

                all_embeddings.extend(data["embeddings"])

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to embed batch {batch_num} at {endpoint}: {e}")
                raise ConnectionError(
                    f"Embedding container error on batch {batch_num}: {e}. "
                    f"Check Swarm service replicas with 'docker service ls'."
                ) from e

        logger.info(
            f"Completed embedding {len(texts)} chunks. "
            f"Load distributed across {len(self.seen_replicas)} Swarm replica(s): {list(self.seen_replicas)}"
        )
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single search query text.
        """
        endpoint = f"{self.api_url}/embed_query"
        try:
            response = self.session.post(
                endpoint,
                json={"text": text, "normalize": True},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if "replica_id" in data:
                self.seen_replicas.add(data["replica_id"])
            return data["embedding"]
        except requests.exceptions.RequestException as e:
            # Fallback to /embed if /embed_query is unavailable
            try:
                fallback_res = self.session.post(
                    f"{self.api_url}/embed",
                    json={"texts": [text], "normalize": True},
                    timeout=self.timeout
                )
                fallback_res.raise_for_status()
                return fallback_res.json()["embeddings"][0]
            except Exception:
                logger.error(f"Failed to embed query at {endpoint}: {e}")
                raise ConnectionError(
                    f"Embedding container error for query: {e}. Is container on {self.api_url}?"
                ) from e

    def ping(self) -> dict:
        """
        Check health and details of the embedding microservice.
        """
        health_endpoint = f"{self.api_url}/health"
        response = self.session.get(health_endpoint, timeout=5)
        response.raise_for_status()
        return response.json()
