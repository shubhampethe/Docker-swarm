import os
import mimetypes
import logging
from typing import List, Optional
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.config import settings

logger = logging.getLogger(__name__)


class UnstructuredContainerLoader:
    """
    Document loader that uploads files to the containerized Unstructured.io API,
    parses document structure (PDF, DOCX, TXT, HTML, etc.), and produces
    standard LangChain Document objects with rich metadata.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.api_url = api_url or settings.unstructured_url
        self.strategy = strategy or settings.unstructured_strategy
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_file(self, file_path: str) -> List[Document]:
        """
        Uploads document to Unstructured container and returns a list of LangChain Documents.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        logger.info(f"Submitting '{file_name}' ({mime_type}) to Unstructured API at {self.api_url}...")

        with open(file_path, "rb") as f:
            files = {"files": (file_name, f, mime_type)}
            data = {
                "strategy": self.strategy,
                "coordinates": "false"
            }
            headers = {"Accept": "application/json"}

            try:
                response = requests.post(
                    self.api_url,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=180
                )
                response.raise_for_status()
                raw_elements = response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Unstructured API request failed: {e}")
                raise ConnectionError(
                    f"Failed to communicate with Unstructured container at {self.api_url}: {e}"
                ) from e

        return self._convert_elements_to_documents(raw_elements, file_path)

    def _convert_elements_to_documents(
        self,
        elements: List[dict],
        source_path: str
    ) -> List[Document]:
        """
        Converts Unstructured raw elements into chunked LangChain Documents.
        """
        file_name = os.path.basename(source_path)
        documents: List[Document] = []

        # Aggregate elements into coherent section blocks
        current_section = ""
        current_metadata = {
            "source": file_name,
            "source_path": source_path,
            "page_number": 1,
            "element_types": []
        }

        for element in elements:
            text = (element.get("text") or "").strip()
            if not text:
                continue

            elem_type = element.get("type", "Unknown")
            meta = element.get("metadata", {})
            page_num = meta.get("page_number", 1)

            # When encountering a major Title or Header, flush previous section
            if elem_type in ["Title", "Header"] and current_section:
                doc = Document(
                    page_content=current_section.strip(),
                    metadata={
                        "source": file_name,
                        "source_path": source_path,
                        "page_number": current_metadata["page_number"],
                        "element_types": ",".join(set(current_metadata["element_types"]))
                    }
                )
                documents.append(doc)
                current_section = ""
                current_metadata["element_types"] = []

            # Append text to section
            current_section += f"\n\n{text}" if current_section else text
            current_metadata["page_number"] = page_num
            current_metadata["element_types"].append(elem_type)

        # Flush final section
        if current_section.strip():
            documents.append(
                Document(
                    page_content=current_section.strip(),
                    metadata={
                        "source": file_name,
                        "source_path": source_path,
                        "page_number": current_metadata["page_number"],
                        "element_types": ",".join(set(current_metadata["element_types"]))
                    }
                )
            )

        # If sections are larger than chunk_size, split them using RecursiveCharacterTextSplitter
        chunked_documents = self.splitter.split_documents(documents)
        logger.info(
            f"Parsed {len(elements)} raw elements from '{file_name}' -> "
            f"{len(documents)} sections -> {len(chunked_documents)} chunks."
        )
        return chunked_documents

    def ping(self) -> bool:
        """
        Check connectivity with Unstructured.io container.
        """
        try:
            # Unstructured API responds to GET on /general/v0/general or root
            res = requests.get(self.api_url, timeout=5)
            # Both 200 and 405 (Method Not Allowed for GET) confirm the endpoint is alive
            return res.status_code in [200, 405]
        except Exception:
            return False
