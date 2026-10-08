import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    # 1. LlamaIndex / LlamaParse Document Parsing Configuration
    llama_cloud_api_key: str = os.getenv("LLAMA_CLOUD_API_KEY", os.getenv("LLAMA_PARSE_API_KEY", ""))
    llama_parse_result_type: str = os.getenv("LLAMA_PARSE_RESULT_TYPE", "markdown")
    llama_parse_language: str = os.getenv("LLAMA_PARSE_LANGUAGE", "en")

    # 2. Custom Embedding Service Container
    embedding_api_url: str = os.getenv("EMBEDDING_API_URL", "http://localhost:8001")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))

    # 3. Local Persistent Chroma Vector Store
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

    # 4. Chunking & Retrieval Parameters
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))

    # 5. LLM Synthesis Options: 'groq', 'openai', 'ollama', 'mock'
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")

    # 6. Docker Swarm Automatic Scaling Configuration
    auto_scale_swarm: bool = os.getenv("AUTO_SCALE_SWARM", "true").lower() in ["true", "1", "yes"]
    swarm_service_name: str = os.getenv("SWARM_SERVICE_NAME", "rag_stack_embedding-service")
    swarm_scale_threshold: int = int(os.getenv("SWARM_SCALE_THRESHOLD", "50"))
    swarm_scale_up_replicas: int = int(os.getenv("SWARM_SCALE_UP_REPLICAS", "3"))
    swarm_idle_replicas: int = int(os.getenv("SWARM_IDLE_REPLICAS", "1"))


settings = Settings()
