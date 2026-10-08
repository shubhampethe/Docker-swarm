import sys
import argparse

# Force UTF-8 encoding on Windows terminals to support non-ASCII markdown quotes/hyphens
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from rag.pipeline import RAGPipeline
from rag.config import settings

# Suppress "Exception ignored while finalizing file <rich.file_proxy.FileProxy>" on exit
def _suppress_rich_shutdown_exception(unraisable):
    if unraisable.exc_type is ImportError and "sys.meta_path" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)

sys.unraisablehook = _suppress_rich_shutdown_exception

console = Console(force_terminal=True)


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]LangChain + LlamaParse RAG Pipeline[/bold cyan]\n"
            "[green]• Parser: LlamaIndex LlamaParse (High-Accuracy Document Parsing)[/green]\n"
            "[green]• Container: Custom Embedding Service (Docker Port 8001 | all-MiniLM-L6-v2)[/green]\n"
            "[green]• Vector DB: Chroma (Local Persistent Data Store)[/green]",
            border_style="cyan"
        )
    )


def cmd_status(pipeline: RAGPipeline):
    console.print("\n[bold yellow]Checking pipeline component status...[/bold yellow]\n")
    health = pipeline.health_check()

    table = Table(title="Pipeline Component Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Endpoint / Mode", style="white", width=38)
    table.add_column("Status", width=16)
    table.add_column("Details", style="dim", width=28)

    # 1. LlamaParse
    lp = health["llamaparse_parser"]
    lp_status = "[bold green]Configured[/bold green]" if lp["configured"] else "[yellow]Fallback[/yellow]"
    lp_details = f"Mode: {lp['status']} | Type: {lp['result_type']}"
    table.add_row("LlamaParse (LlamaIndex)", "Llama Cloud API", lp_status, lp_details)

    # 2. Embedding Service Container
    emb = health["embedding_container"]
    emb_status = "[bold green]Online[/bold green]" if emb["reachable"] else "[bold red]Unreachable[/bold red]"
    emb_details = f"Model: {emb.get('details', {}).get('model', 'N/A')}" if emb["reachable"] else "Check docker container (port 8001)"
    table.add_row("Embedding Container", emb["url"], emb_status, emb_details)

    # 3. Chroma Local
    chr_s = health["chroma_local_store"]
    chr_status = "[bold green]Ready[/bold green]" if chr_s["ready"] else "[bold red]Error[/bold red]"
    chr_details = f"Docs: {chr_s['total_documents']} | Col: {chr_s['collection']}"
    table.add_row("Chroma Vector Store", f"Directory: {chr_s['persist_directory']}", chr_status, chr_details)

    console.print(table)


def cmd_ingest(pipeline: RAGPipeline, file_path: str):
    console.print(f"\n[bold cyan]Ingesting file:[/bold cyan] [yellow]{file_path}[/yellow]")
    with console.status("[bold green]Parsing with LlamaParse -> Embedding via Container -> Storing in Chroma...[/bold green]"):
        try:
            result = pipeline.ingest_file(file_path)
            console.print(Panel(
                f"[bold green]Ingestion Complete![/bold green]\n"
                f"• File: {result.get('file')}\n"
                f"• Chunks Created: {result.get('chunks_created')}\n"
                f"• Chunks Indexed: {result.get('chunks_indexed')}\n"
                f"• Total Collection Size: {result.get('total_collection_count')}",
                title="Success",
                border_style="green"
            ))
        except Exception as e:
            console.print(Panel(f"[bold red]Ingestion Failed:[/bold red] {e}", title="Error", border_style="red"))


def cmd_query(pipeline: RAGPipeline, question: str, top_k: int = 4):
    console.print(f"\n[bold cyan]Query:[/bold cyan] [white]{question}[/white]\n")
    with console.status("[bold green]Searching Chroma & Synthesizing Answer...[/bold green]"):
        try:
            result = pipeline.query(question, top_k=top_k)
        except Exception as e:
            console.print(Panel(f"[bold red]Query Execution Failed:[/bold red] {e}", title="Error", border_style="red"))
            return

    # Print Answer
    console.print(Panel(
        Markdown(result["answer"]),
        title="[bold green]RAG Response[/bold green]",
        border_style="green"
    ))

    # Print Citations
    if result["sources"]:
        table = Table(title="Retrieved Citations & Sources", show_header=True, header_style="bold blue")
        table.add_column("#", width=4)
        table.add_column("Document", width=22)
        table.add_column("Page", width=6)
        table.add_column("Content Snippet", style="dim")

        for idx, src in enumerate(result["sources"], start=1):
            table.add_row(str(idx), src["source"], str(src["page"]), src["preview"])

        console.print(table)


def cmd_search(pipeline: RAGPipeline, query_text: str, top_k: int = 4):
    console.print(f"\n[bold cyan]Direct Vector Search:[/bold cyan] [white]{query_text}[/white]\n")
    try:
        results = pipeline.search_similar(query_text, top_k=top_k)
        if not results:
            console.print("[yellow]No matching chunks found.[/yellow]")
            return

        for idx, (doc, score) in enumerate(results, start=1):
            console.print(Panel(
                f"[bold cyan]Source:[/bold cyan] {doc.metadata.get('source')} | [bold cyan]Page:[/bold cyan] {doc.metadata.get('page_number')} | [bold cyan]Score:[/bold cyan] {score:.4f}\n\n"
                f"{doc.page_content}",
                title=f"Result #{idx}",
                border_style="blue"
            ))
    except Exception as e:
        console.print(f"[bold red]Search error:[/bold red] {e}")


def cmd_interactive(pipeline: RAGPipeline):
    console.print("\n[bold green]Interactive RAG Mode. Type your question or 'exit'/'quit' to leave.[/bold green]\n")
    while True:
        try:
            question = console.input("[bold cyan]Ask > [/bold cyan]").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break
            cmd_query(pipeline, question)
            console.print("\n" + "-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break


def main():
    parser = argparse.ArgumentParser(description="LangChain + LlamaParse Containerized RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status
    subparsers.add_parser("check", help="Check status of LlamaParse, Embedding container, and Chroma store")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a document file")
    ingest_parser.add_argument("file_path", help="Path to PDF, DOCX, TXT, or markdown file")

    # Query
    query_parser = subparsers.add_parser("query", help="Ask a question against indexed documents")
    query_parser.add_argument("question", help="Question text")
    query_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")

    # Search
    search_parser = subparsers.add_parser("search", help="Direct semantic similarity search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve")

    # Interactive
    subparsers.add_parser("interactive", help="Start interactive Q&A session")

    # Test 100 Chunks Flow
    subparsers.add_parser("test-100", help="Run 100+ chunk ingestion and verify Swarm replica distribution")

    args = parser.parse_args()

    print_banner()
    pipeline = RAGPipeline()

    if args.command == "check":
        cmd_status(pipeline)
    elif args.command == "ingest":
        cmd_ingest(pipeline, args.file_path)
    elif args.command == "query":
        cmd_query(pipeline, args.question, top_k=args.top_k)
    elif args.command == "search":
        cmd_search(pipeline, args.query, top_k=args.top_k)
    elif args.command == "interactive":
        cmd_interactive(pipeline)
    elif args.command == "test-100":
        console.print("\n[bold cyan]Running 100+ Chunk Swarm Verification Benchmark...[/bold cyan]")
        cmd_ingest(pipeline, "sample_docs/large_knowledge_base.txt")
        replicas = list(pipeline.embeddings.seen_replicas)
        console.print(Panel(
            f"[bold green]Swarm Load Balancing Summary[/bold green]\n"
            f"• Replicas handling requests: [cyan]{len(replicas)}[/cyan]\n"
            f"• Replica IDs: [white]{', '.join(replicas) if replicas else 'localhost'}[/white]",
            title="Swarm Distribution",
            border_style="green"
        ))
        cmd_query(pipeline, "What is the consensus algorithm used in distributed systems?")
    else:
        cmd_status(pipeline)
        console.print("\n[yellow]Tip: Run with 'python main.py --help' to view available commands.[/yellow]")


if __name__ == "__main__":
    main()
