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
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = ""  # if empty, falls back to embedding_model

    # Retrieval defaults
    default_max_hop: int = 2
    default_top_k: int = 4
    # Graph traversal mode:
    #   "distinct"   — distinct relationships within hop, LIMIT 500 (default;
    #                  no path-explosion truncation, recall == undirected on eval)
    #   "undirected" — paths in both directions, RETURN p LIMIT 200
    #   "directed"   — outgoing-only paths, RETURN p LIMIT 200 (lower recall)
    graph_traversal_mode: str = "distinct"

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
