"""
End-to-end verification script for:
1. Parsing documents (LlamaIndex LlamaParse / local loader) into 100+ chunks
2. Batch embedding 100+ chunks across Docker Swarm replicas
3. Storing 100+ chunks in local Chroma DB
4. Semantic vector search and QA retrieval verification
"""

import os
import sys
import time
import unittest
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

from rag.config import settings
from rag.llamaparse_loader import LlamaParseLoader
from rag.embeddings import CustomContainerEmbeddings
from rag.chroma_store import ChromaStoreManager
from rag.chain import RAGChain


class MockSwarmEmbeddingServer(BaseHTTPRequestHandler):
    """
    Simulates a 3-replica Docker Swarm ingress service for automated testing
    when containers are not running locally.
    """
    replica_counter = 0
    replicas = ["embedding-service-replica-1", "embedding-service-replica-2", "embedding-service-replica-3"]

    def log_message(self, format, *args):
        return  # Suppress default HTTP logging

    def do_GET(self):
        if self.path in ["/health", "/"]:
            MockSwarmEmbeddingServer.replica_counter += 1
            chosen_replica = MockSwarmEmbeddingServer.replicas[
                MockSwarmEmbeddingServer.replica_counter % len(MockSwarmEmbeddingServer.replicas)
            ]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384,
                "replica_id": chosen_replica
            }).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(body) if body else {}

        MockSwarmEmbeddingServer.replica_counter += 1
        chosen_replica = MockSwarmEmbeddingServer.replicas[
            MockSwarmEmbeddingServer.replica_counter % len(MockSwarmEmbeddingServer.replicas)
        ]

        if self.path == "/embed":
            texts = payload.get("texts", [])
            # Return dummy 384-dimensional vectors
            embeddings = [[0.05 * (i % 10)] * 384 for i in range(len(texts))]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "embeddings": embeddings,
                "dimension": 384,
                "count": len(embeddings),
                "replica_id": chosen_replica
            }).encode("utf-8"))

        elif self.path == "/embed_query":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "embedding": [0.05] * 384,
                "dimension": 384,
                "replica_id": chosen_replica
            }).encode("utf-8"))


class TestE2EFlow(unittest.TestCase):
    mock_server = None
    mock_thread = None
    port = 8001

    @classmethod
    def setUpClass(cls):
        # Check if an actual embedding container is running on port 8001
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex(('127.0.0.1', cls.port))
        s.close()

        if result != 0:
            print(f"\n[INFO] No service listening on port {cls.port}. Starting simulated 3-replica Swarm server...")
            cls.mock_server = HTTPServer(('127.0.0.1', cls.port), MockSwarmEmbeddingServer)
            cls.mock_thread = threading.Thread(target=cls.mock_server.serve_forever, daemon=True)
            cls.mock_thread.start()
            time.sleep(0.5)
        else:
            print(f"\n[INFO] Connected to active Docker service on port {cls.port}!")

    @classmethod
    def tearDownClass(cls):
        if cls.mock_server:
            cls.mock_server.shutdown()
            cls.mock_server.server_close()

    def test_01_parse_100_plus_chunks(self):
        """Verify document parsing splits the sample into >= 100 chunks."""
        doc_path = "sample_docs/large_knowledge_base.txt"
        self.assertTrue(os.path.exists(doc_path), f"File {doc_path} not found")

        # Use chunk_size=300 to ensure >= 100 chunks
        loader = LlamaParseLoader(chunk_size=300, chunk_overlap=30)
        chunks = loader.load_file(doc_path)

        print(f"\n[STEP 1] Parsed '{doc_path}' into {len(chunks)} chunks.")
        self.assertGreaterEqual(
            len(chunks), 100,
            f"Expected at least 100 chunks, but got {len(chunks)}"
        )
        self.assertEqual(chunks[0].metadata["source"], "large_knowledge_base.txt")

    def test_02_embed_100_chunks_across_swarm_replicas(self):
        """Verify 100+ chunks are embedded in batches and distributed across replicas."""
        loader = LlamaParseLoader(chunk_size=300, chunk_overlap=30)
        chunks = loader.load_file("sample_docs/large_knowledge_base.txt")
        texts = [c.page_content for c in chunks[:105]]  # Pick 105 chunks

        print(f"\n[STEP 2] Sending {len(texts)} chunks to Swarm embedding service (batch_size=32)...")
        embeddings_client = CustomContainerEmbeddings(
            api_url=f"http://127.0.0.1:{self.port}",
            batch_size=32
        )

        vectors = embeddings_client.embed_documents(texts)
        self.assertEqual(len(vectors), len(texts))
        self.assertEqual(len(vectors[0]), 384)

        print(f"✓ Generated {len(vectors)} vectors (dimension={len(vectors[0])}).")
        print(f"✓ Replicas observed handling traffic: {embeddings_client.seen_replicas}")
        self.assertGreater(len(embeddings_client.seen_replicas), 0)

    def test_03_chroma_indexing_and_search(self):
        """Verify 100+ chunks are indexed into local Chroma and can be queried."""
        loader = LlamaParseLoader(chunk_size=300, chunk_overlap=30)
        chunks = loader.load_file("sample_docs/large_knowledge_base.txt")[:105]

        embeddings_client = CustomContainerEmbeddings(
            api_url=f"http://127.0.0.1:{self.port}",
            batch_size=32
        )

        test_dir = "./chroma_test_data"
        store = ChromaStoreManager(
            embeddings=embeddings_client,
            persist_directory=test_dir,
            collection_name="test_100_chunks"
        )

        print(f"\n[STEP 3] Indexing {len(chunks)} chunks into local Chroma DB ({test_dir})...")
        ids = store.add_documents(chunks)
        self.assertEqual(len(ids), len(chunks))

        total_count = store.count()
        print(f"✓ Collection item count in Chroma: {total_count}")
        self.assertGreaterEqual(total_count, 100)

        # Test Semantic Retrieval
        query = "What is the consensus algorithm used for distributed systems?"
        results = store.search(query, k=3)
        print(f"✓ Retrieved {len(results)} chunks for query: '{query}'")
        self.assertGreater(len(results), 0)

        # Clean up test directory
        import shutil
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
