import os
import socket
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("embedding-service")

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
HOSTNAME = socket.gethostname()

app = FastAPI(
    title="Containerized Embedding Service",
    description="FastAPI service for generating text embeddings using Sentence-Transformers with Docker Swarm scale support",
    version="1.1.0"
)

logger.info(f"Loading embedding model: {MODEL_NAME} on replica: {HOSTNAME}...")
model = SentenceTransformer(MODEL_NAME)
embedding_dim = model.get_sentence_embedding_dimension()
logger.info(f"Model loaded successfully on {HOSTNAME}. Dimension: {embedding_dim}")


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text chunks to embed")
    normalize: bool = Field(default=True, description="Whether to L2-normalize vectors")


class EmbedQueryRequest(BaseModel):
    text: str = Field(..., description="Single query string to embed")
    normalize: bool = Field(default=True, description="Whether to L2-normalize vector")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int
    replica_id: str


class EmbedQueryResponse(BaseModel):
    embedding: List[float]
    dimension: int
    replica_id: str


@app.get("/")
def root():
    return {
        "service": "Embedding Service",
        "model": MODEL_NAME,
        "dimension": embedding_dim,
        "replica_id": HOSTNAME,
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": embedding_dim,
        "replica_id": HOSTNAME
    }


@app.post("/embed", response_model=EmbedResponse)
def embed_documents(request: EmbedRequest):
    if not request.texts:
        return EmbedResponse(embeddings=[], dimension=embedding_dim, count=0, replica_id=HOSTNAME)
    try:
        vectors = model.encode(
            request.texts,
            normalize_embeddings=request.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings_list = vectors.tolist()
        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=embedding_dim,
            count=len(embeddings_list),
            replica_id=HOSTNAME
        )
    except Exception as e:
        logger.error(f"Error generating embeddings on {HOSTNAME}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_query", response_model=EmbedQueryResponse)
def embed_query(request: EmbedQueryRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    try:
        vector = model.encode(
            request.text,
            normalize_embeddings=request.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return EmbedQueryResponse(
            embedding=vector.tolist(),
            dimension=embedding_dim,
            replica_id=HOSTNAME
        )
    except Exception as e:
        logger.error(f"Error generating query embedding on {HOSTNAME}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
