from pathlib import Path
from dotenv import load_dotenv
import os

from llama_index.core import Settings
from llama_index.core.base.llms.types import LLMMetadata, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value


def get_required_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_str_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def get_int_env(name: str, default: int) -> int:
    value = get_env(name)
    return int(value) if value else default


LLM_PROVIDER = get_str_env("LLM_PROVIDER", "openai")
EMBED_PROVIDER = get_str_env("EMBED_PROVIDER", "openai")

LLM_MODEL = get_required_env("LLM_MODEL")
EMBED_MODEL = get_required_env("EMBED_MODEL")
LLM_CONTEXT_WINDOW = get_int_env("LLM_CONTEXT_WINDOW", 65536)
LLM_NUM_OUTPUT = get_int_env("LLM_NUM_OUTPUT", 4096)
CHUNK_SIZE = get_int_env("CHUNK_SIZE", 2048)
CHUNK_OVERLAP = get_int_env("CHUNK_OVERLAP", 128)

DEEPSEEK_API_KEY = get_env("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = get_str_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

OPENAI_API_KEY = get_env("OPENAI_API_KEY")

DATA_DIR = PROJECT_ROOT / get_str_env("DATA_DIR", "data")
STORAGE_DIR = PROJECT_ROOT / get_str_env("STORAGE_DIR", "storage")
MANIFEST_PATH = PROJECT_ROOT / get_str_env("MANIFEST_PATH", "manifest.yaml")


class OpenAICompatibleLLM(OpenAI):
    """OpenAI-compatible chat API wrapper for non-OpenAI model names."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=LLM_CONTEXT_WINDOW,
            num_output=self.max_tokens or LLM_NUM_OUTPUT,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model,
            system_role=MessageRole.SYSTEM,
        )

    @property
    def _tokenizer(self):
        return None


def setup_llm():
    if LLM_PROVIDER == "deepseek":
        return OpenAICompatibleLLM(
            model=LLM_MODEL,
            api_key=DEEPSEEK_API_KEY,
            api_base=DEEPSEEK_BASE_URL,
            max_tokens=LLM_NUM_OUTPUT,
            additional_kwargs={
                "extra_body": {
                    "thinking": {"type": "enabled"}
                }
            },
        )

    if LLM_PROVIDER == "openai":
        return OpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def setup_embed_model():
    if EMBED_PROVIDER == "openai":
        return OpenAIEmbedding(
            model=EMBED_MODEL,
            api_key=OPENAI_API_KEY,
        )

    raise RuntimeError(f"Unsupported EMBED_PROVIDER: {EMBED_PROVIDER}")

##############################################################################
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "rag_pymupdf_v1"
EMBED_DIM = 1536
##############################################################################
Settings.llm = setup_llm()
Settings.embed_model = setup_embed_model()
Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP
