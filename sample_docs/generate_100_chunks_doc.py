import os

TOPICS = [
    ("Distributed Systems", "Consensus algorithms such as Raft and Paxos ensure consistency across distributed nodes by establishing leader leases and state machine replication."),
    ("Vector Indexing", "Hierarchical Navigable Small World (HNSW) graphs allow sub-linear approximate nearest neighbor vector search with high recall in Chroma."),
    ("API Gateway Design", "Kong and Envoy act as reverse proxies handling TLS termination, JWT verification, rate limiting, and canary traffic routing."),
    ("Polyglot Persistence", "Combining relational databases for ACID transactions, document stores for schemaless payloads, and vector stores for embeddings optimizes throughput."),
    ("Event-Driven Architecture", "Kafka message brokers ingest event logs into partitioned topics, allowing consumer groups to process order events asynchronously."),
    ("Caching Patterns", "Cache-aside and write-through caching using Redis Cluster minimizes latency for hot product listings while preventing cache stampede."),
    ("Disaster Recovery", "Multi-region active-passive deployments achieve Recovery Point Objective (RPO) under 1 second and Recovery Time Objective (RTO) under 30 seconds."),
    ("Container Orchestration", "Docker Swarm ingress routing mesh load balances incoming HTTP traffic round-robin across replicated worker service tasks."),
    ("Observability & Tracing", "OpenTelemetry collectors aggregate distributed traces, Prometheus metrics, and Loki log streams into centralized Grafana dashboards."),
    ("Database Sharding", "Consistent hashing partitions database write operations across multiple primary database shards without centralized coordination bottlenecks.")
]

def generate_document(output_path: str, target_sections: int = 60):
    lines = [
        "# Enterprise Distributed Cloud Architecture & Knowledge Base",
        "## Overview",
        "This enterprise manual provides comprehensive specifications for high-throughput distributed microservices, vector search indexing, and resilient cloud infrastructure.",
        ""
    ]

    count = 0
    while count < target_sections:
        for idx, (title, description) in enumerate(TOPICS, start=1):
            count += 1
            lines.append(f"## Section {count}: {title} (Module {idx})")
            lines.append(f"Specification Code: SPEC-{count:04d}")
            lines.append(description)
            lines.append(f"Operational Guideline: Engineers must ensure module {count} maintains 99.999% availability and complies with ISO-27001 standard protocols.")
            lines.append(f"Technical parameter benchmark: Latency threshold is 15ms under a baseline load of {count * 1500} concurrent active sessions.\n")
            if count >= target_sections:
                break

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated {count} sections in {output_path} ({len(content)} characters).")

if __name__ == "__main__":
    generate_document("sample_docs/large_knowledge_base.txt", target_sections=70)
