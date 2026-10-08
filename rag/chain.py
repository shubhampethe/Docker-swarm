import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.config import settings

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """You are a helpful, precise AI assistant. Answer the user's question using ONLY the retrieved context below.
If the answer cannot be determined from the context, state that clearly. Do not make up facts.

---------------------
Retrieved Context:
{context}
---------------------

User Question: {question}

Helpful Answer:"""


def format_docs_with_sources(docs: List[Document]) -> str:
    """
    Format a list of documents into a labeled context block.
    """
    if not docs:
        return "No relevant context found."

    formatted_blocks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page_number", 1)
        elem_types = doc.metadata.get("element_types", "Text")
        header = f"[Chunk {idx} | Source: {source} (Page {page}) | Types: {elem_types}]"
        formatted_blocks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted_blocks)


class FallbackExtractiveLLM:
    """
    Lightweight local extractive generator that constructs an answer from retrieved chunks
    without requiring external paid API keys or cloud services.
    """

    def invoke(self, prompt_text: str) -> str:
        # Extract question and context from the prompt
        lines = prompt_text.split("\n")
        context_lines = []
        question = ""
        in_context = False

        for line in lines:
            if "Retrieved Context:" in line:
                in_context = True
                continue
            if "User Question:" in line:
                in_context = False
                question = line.replace("User Question:", "").strip()
                continue
            if in_context and not line.startswith("---"):
                context_lines.append(line)

        context_body = "\n".join(context_lines).strip()
        if not context_body or "No relevant context found" in context_body:
            return "No matching information was found in the indexed documents for this question."

        # Provide a clean grounded summary
        return (
            f"Based on the retrieved document context:\n\n"
            f"{context_body}\n\n"
            f"(Note: Running with local extractive synthesizer. Set GROQ_API_KEY in .env to use ChatGroq)."
        )


def get_llm():
    """
    Initializes the configured LLM or returns the local fallback.
    """
    provider = settings.llm_provider.lower()

    if provider == "groq" and settings.groq_api_key:
        try:
            from langchain_groq import ChatGroq
            logger.info(f"Using ChatGroq LLM with model: {settings.groq_model}")
            return ChatGroq(
                model=settings.groq_model,
                groq_api_key=settings.groq_api_key,
                temperature=0.2
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChatGroq ({e}). Falling back to local synthesizer.")

    elif provider == "openai" and settings.openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=0.2
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChatOpenAI ({e}). Falling back to local synthesizer.")

    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                temperature=0.2
            )
        except Exception as e:
            logger.warning(f"Could not initialize ChatOllama ({e}). Falling back to local synthesizer.")

    return FallbackExtractiveLLM()


class RAGChain:
    """
    Assembles the LangChain retrieval and answer generation workflow.
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = get_llm()
        self.prompt = PromptTemplate(
            template=RAG_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Executes end-to-end retrieval and generation for a question.
        """
        # 1. Retrieve relevant chunks
        docs: List[Document] = self.retriever.invoke(question)

        # 2. Format context
        context_str = format_docs_with_sources(docs)

        # 3. Format prompt
        rendered_prompt = self.prompt.format(context=context_str, question=question)

        # 4. Synthesize answer
        if hasattr(self.llm, "invoke"):
            raw_response = self.llm.invoke(rendered_prompt)
            answer_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        else:
            answer_text = str(self.llm(rendered_prompt))

        # 5. Extract citations/sources
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page_number", 1),
                "element_types": doc.metadata.get("element_types", "Text"),
                "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            }
            for doc in docs
        ]

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "chunks_retrieved": len(docs),
            "raw_context": context_str
        }
