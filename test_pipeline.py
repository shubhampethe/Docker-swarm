"""
Automated validation and integration tests for the RAG Pipeline.
"""

import os
import unittest
from langchain_core.documents import Document
from rag.unstructured_loader import UnstructuredContainerLoader
from rag.chain import format_docs_with_sources, FallbackExtractiveLLM, RAGChain


class TestRAGComponents(unittest.TestCase):

    def test_element_to_document_conversion(self):
        """Test Unstructured raw element to LangChain Document conversion."""
        loader = UnstructuredContainerLoader(chunk_size=200, chunk_overlap=20)
        sample_elements = [
            {
                "type": "Title",
                "text": "System Overview",
                "metadata": {"page_number": 1}
            },
            {
                "type": "NarrativeText",
                "text": "This is a high performance RAG pipeline running in Docker.",
                "metadata": {"page_number": 1}
            },
            {
                "type": "Title",
                "text": "Database Tier",
                "metadata": {"page_number": 2}
            },
            {
                "type": "NarrativeText",
                "text": "Chroma vector database handles document indexing and similarity search.",
                "metadata": {"page_number": 2}
            }
        ]

        docs = loader._convert_elements_to_documents(sample_elements, "sample_docs/sample.txt")
        self.assertGreaterEqual(len(docs), 2)
        self.assertIn("System Overview", docs[0].page_content)
        self.assertEqual(docs[0].metadata["source"], "sample.txt")

    def test_format_docs_with_sources(self):
        """Test document formatting with citation metadata."""
        docs = [
            Document(
                page_content="RTO is 30 seconds and RPO is under 1 second.",
                metadata={"source": "disaster_recovery.pdf", "page_number": 4, "element_types": "Text"}
            )
        ]
        formatted = format_docs_with_sources(docs)
        self.assertIn("Source: disaster_recovery.pdf", formatted)
        self.assertIn("Page 4", formatted)
        self.assertIn("RTO is 30 seconds", formatted)

    def test_fallback_extractive_synthesizer(self):
        """Test local fallback synthesizer when no external LLM API key is present."""
        llm = FallbackExtractiveLLM()
        prompt = (
            "Retrieved Context:\n"
            "[Chunk 1 | Source: test.txt (Page 1)]\n"
            "Chroma vector database is used for item embeddings.\n\n"
            "User Question: What database is used for item embeddings?\n\n"
            "Helpful Answer:"
        )
        response = llm.invoke(prompt)
        self.assertIn("Chroma vector database", response)


if __name__ == "__main__":
    unittest.main()
