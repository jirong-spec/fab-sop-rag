from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (OpenAI-compatible endpoint)
    openai_api_key: str = "EMPTY"
    # Default uses Docker service name; override to http://localhost:8299/v1 for local dev
    openai_api_base: str = "http://vllm:8299/v1"
    llm_model: str = "Qwen2.5-3B-Instruct"
    llm_timeout: float = 30.0

    # Neo4j
    # Default uses Docker service name; override to bolt://localhost:7687 for local dev
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password123"

    # Vector store (Chroma)
    # Default is the Docker volume mount path.
    # For local dev without Docker: CHROMA_DIR=../lab1/chroma_store
    chroma_dir: str = "/data/chroma"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

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
