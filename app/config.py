from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "1.0.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (OpenAI-compatible endpoint)
    openai_api_key: str = "EMPTY"
    # Default uses Docker service name (container port 8000).
    # For local dev outside Docker: override to http://localhost:8299/v1
    openai_api_base: str = "http://vllm:8000/v1"
    llm_model: str = "Qwen2.5-7B-Instruct-AWQ-int4"
    llm_timeout: float = 30.0
    # Model context window (must match vLLM --max-model-len). The prompt builder
    # trims retrieved triples to fit this budget so a dense graph can't overflow it.
    llm_max_model_len: int = 4096

    # Neo4j
    # Default uses Docker service name; override to bolt://localhost:7687 for local dev
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password123"

    # Vector store (Qdrant)
    # Default uses the Docker service name; override to http://localhost:6333 for local dev.
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "sop_docs"
    # Doc-retrieval embedder. e5-small chosen via scripts/eval_chunk_ablation-style bake-off:
    # on held-out retrieval it beat the prior MiniLM across recall@4/kw-RR/nDCG (directional,
    # not 95%-significant on 19 q); picked over the tied gte-multilingual-base purely for VRAM
    # (same 384-dim, no extra cost) + standard arch. See data/eval_results/ + interview report.
    embedding_model: str = "intfloat/multilingual-e5-small"
    # Reranker pinned to the small MiniLM (not the doc embedder) to avoid loading e5 twice (VRAM)
    # and because triple reranking does not need e5's query/passage prefixes.
    reranker_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # e5-family instruction prefixes, applied at embed time only (stored text stays clean).
    # Leave empty for non-e5 models (MiniLM, gte, ...).
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # Retrieval defaults
    default_max_hop: int = 2
    default_top_k: int = 4

    # Guardrail fallback policies
    # topic_fallback_policy:     "lenient" = allow + warn | "strict" = block
    # grounding_fallback_policy: "strict"  = flag ungrounded | "lenient" = allow + warn
    topic_fallback_policy: str = "lenient"
    grounding_fallback_policy: str = "strict"

    # Authentication
    # Set to a non-empty string to enable X-API-Key header validation.
    # Leave empty to disable auth (suitable for internal demos).
    api_key: str = ""


settings = Settings()
