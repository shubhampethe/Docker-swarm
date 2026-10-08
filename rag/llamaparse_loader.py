import os
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.config import settings

logger = logging.getLogger(__name__)


def _run_llamaparse_in_thread(api_key: str, result_type: str, language: str, file_path: str) -> list:
    """
    Runs LlamaParse inside a COMPLETELY ISOLATED thread with its OWN fresh event loop.

    Root cause of 'unknown async library / not in async context':
        LlamaParse uses sniffio/anyio to auto-detect the async backend. When called from
        a synchronous CLI or Streamlit context, there is no recognized anyio-managed loop
        in scope, so sniffio raises the error regardless of nest_asyncio patches.

    Fix: Spin up a dedicated daemon thread, create a brand-new asyncio event loop inside it,
        and run LlamaParse exclusively there. This is completely isolated from the caller's
        thread and loop context.
    """
    result: List = []
    error: List = []

    def _thread_target():
        # Create and set a FRESH event loop in this thread (critical)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from llama_parse import LlamaParse
            parser = LlamaParse(
                api_key=api_key,
                result_type=result_type,
                language=language,
                verbose=False
            )
            docs = loop.run_until_complete(parser.aload_data(file_path))
            result.extend(docs)
        except Exception as e:
            error.append(e)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_thread_target, daemon=True)
    t.start()
    t.join(timeout=300)  # 5-minute timeout for large PDFs

    if error:
        raise error[0]
    return result


class LlamaParseLoader:
    """
    Document loader using LlamaIndex's LlamaParse service to parse complex documents
    (PDF, PPTX, DOCX, TXT, HTML) into structured markdown, and convert them into
    chunked LangChain Document objects.

    Parsing is executed in a thread-isolated environment with a fresh asyncio event loop
    to avoid sniffio/anyio 'unknown async library' conflicts when called from CLI,
    Streamlit, or any synchronous context.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: Optional[str] = None,
        language: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.api_key = api_key or settings.llama_cloud_api_key
        self.result_type = result_type or settings.llama_parse_result_type
        self.language = language or settings.llama_parse_language
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_file(self, file_path: str) -> List[Document]:
        """
        Loads and parses a document via LlamaParse (thread-isolated async execution)
        or falls back to local reader if API key is not set.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)

        # 1. Use LlamaParse if API key is configured
        if self.api_key and not self.api_key.startswith("your_"):
            try:
                logger.info(
                    f"Submitting '{file_name}' to LlamaParse "
                    f"(result_type={self.result_type}) via thread-isolated async runner..."
                )
                # Run in isolated daemon thread with its own clean event loop
                llama_docs = _run_llamaparse_in_thread(
                    api_key=self.api_key,
                    result_type=self.result_type,
                    language=self.language,
                    file_path=file_path
                )

                raw_documents: List[Document] = []
                for idx, l_doc in enumerate(llama_docs, start=1):
                    text_content = l_doc.text if hasattr(l_doc, "text") else str(l_doc)
                    metadata = {
                        "source": file_name,
                        "source_path": file_path,
                        "page_number": getattr(l_doc, "metadata", {}).get("page_number", idx),
                        "parser": "llama-parse",
                        **(getattr(l_doc, "metadata", {}) or {})
                    }
                    raw_documents.append(Document(page_content=text_content, metadata=metadata))

                chunked = self.splitter.split_documents(raw_documents)
                logger.info(
                    f"LlamaParse parsed {len(llama_docs)} pages from '{file_name}' -> {len(chunked)} chunks."
                )
                return chunked

            except Exception as e:
                logger.error(f"LlamaParse error: {e}. Falling back to local loader.")

        # 2. Local fallback reader (plain text/markdown, no API key needed)
        logger.warning(
            f"LlamaParse API key not configured or call failed. "
            f"Using local fallback reader for '{file_name}'."
        )
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            raw_doc = Document(
                page_content=content,
                metadata={
                    "source": file_name,
                    "source_path": file_path,
                    "page_number": 1,
                    "parser": "local_fallback"
                }
            )
            chunked = self.splitter.split_documents([raw_doc])
            logger.info(f"Local loader extracted {len(chunked)} chunks from '{file_name}'.")
            return chunked
        except Exception as e:
            raise RuntimeError(f"Failed to read file '{file_path}': {e}") from e

    async def aload_file(self, file_path: str) -> List[Document]:
        """
        Async-compatible wrapper: runs load_file in a ThreadPoolExecutor so it
        doesn't block the event loop when called from Streamlit or other async callers.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load_file, file_path)

    def is_configured(self) -> bool:
        """
        Check if LlamaParse API key is configured.
        """
        return bool(self.api_key and not self.api_key.startswith("your_"))
