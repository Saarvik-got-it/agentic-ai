"""
Embedding utilities using Google Gemini.
Abstracts the embedding provider and provides a reusable interface.
"""

from typing import List
import google.generativeai as genai
from langchain_core.embeddings import Embeddings
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Try both model name formats to handle SDK/API compatibility differences.
EMBEDDING_MODEL_CANDIDATES = [
    "embedding-001",
    "models/embedding-001",
]
EMBEDDING_BATCH_SIZE = 32


class GeminiSDKEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper powered by google.generativeai SDK.
    This keeps FAISS integration unchanged while bypassing LangChain embedding adapters.
    """

    def __init__(self, api_key: str, model_candidates: List[str]):
        self._api_key = api_key
        self._model_candidates = model_candidates
        self._model_name = None

        genai.configure(api_key=self._api_key)
        self._model_name = self._resolve_model()

    def _get_discovered_model_candidates(self) -> List[str]:
        """
        Discover embedding-capable models exposed by the current API key.
        This protects against API-version/model-name drift.
        """
        discovered: List[str] = []
        try:
            for model in genai.list_models():
                methods = getattr(model, "supported_generation_methods", []) or []
                model_name = getattr(model, "name", "")
                if "embedContent" in methods and model_name:
                    discovered.append(model_name)
        except Exception as e:
            logger.warning("Unable to discover embedding models dynamically: %s", str(e))

        return discovered

    @staticmethod
    def _normalize_embeddings(raw_response, expected_count: int) -> List[List[float]]:
        """Normalize SDK response formats into List[List[float]]."""
        embeddings = None

        if isinstance(raw_response, dict):
            if "embedding" in raw_response:
                embeddings = raw_response["embedding"]
            elif "embeddings" in raw_response:
                embeddings = raw_response["embeddings"]
        else:
            if hasattr(raw_response, "embedding"):
                embeddings = getattr(raw_response, "embedding")
            elif hasattr(raw_response, "embeddings"):
                embeddings = getattr(raw_response, "embeddings")

        if embeddings is None:
            raise ValueError("Embed API response missing embedding data")

        # Case: single vector returned for one input.
        if embeddings and isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], (int, float)):
            normalized = [list(embeddings)]
        else:
            normalized = []
            for item in embeddings:
                if isinstance(item, dict) and "values" in item:
                    normalized.append(list(item["values"]))
                elif isinstance(item, list):
                    normalized.append(list(item))
                else:
                    raise ValueError(f"Unsupported embedding item type: {type(item)}")

        if len(normalized) != expected_count:
            raise ValueError(
                f"Embedding count mismatch. Expected {expected_count}, got {len(normalized)}"
            )

        return normalized

    def _embed_batch_with_model(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Embed one batch using a specific model name."""
        if not texts:
            return []

        content = texts if len(texts) > 1 else texts[0]
        response = genai.embed_content(
            model=model_name,
            content=content,
        )
        return self._normalize_embeddings(response, expected_count=len(texts))

    def _resolve_model(self) -> str:
        """Resolve the first working embedding model from candidates."""
        last_error = None

        discovered_candidates = self._get_discovered_model_candidates()
        if discovered_candidates:
            logger.info(
                "Discovered embedding-capable models from API: %s",
                discovered_candidates,
            )

        # Preserve order and de-duplicate candidates.
        all_candidates: List[str] = []
        for candidate in self._model_candidates + discovered_candidates:
            if candidate not in all_candidates:
                all_candidates.append(candidate)

        for model_name in all_candidates:
            try:
                logger.info(f"Testing Gemini embedding model: {model_name}")
                self._embed_batch_with_model(["healthcheck"], model_name=model_name)
                logger.info(f"Using Gemini embedding model: {model_name}")
                return model_name
            except Exception as e:
                last_error = e
                logger.warning(
                    "Gemini embedding model test failed for %s: %s",
                    model_name,
                    str(e),
                )

        raise RuntimeError(
            "Unable to initialize Gemini embeddings with any supported model. "
            f"Tried: {all_candidates}. Last error: {last_error}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts with batch processing support."""
        if not texts:
            return []

        vectors: List[List[float]] = []
        for start_idx in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start_idx:start_idx + EMBEDDING_BATCH_SIZE]
            try:
                vectors.extend(self._embed_batch_with_model(batch, model_name=self._model_name))
            except Exception as e:
                logger.warning(
                    "Batch embedding failed for items %s-%s with model %s: %s. Falling back to single-item embedding.",
                    start_idx,
                    start_idx + len(batch) - 1,
                    self._model_name,
                    str(e),
                )
                # Fallback to per-item embedding if batch response shape differs.
                for text in batch:
                    vectors.extend(self._embed_batch_with_model([text], model_name=self._model_name))

        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        return self.embed_documents([text])[0]


def get_embeddings():
    """
    Initialize and return Google Gemini embeddings instance.
    
    Returns:
        GeminiSDKEmbeddings instance configured with API key
    
    Raises:
        ValueError: If GOOGLE_API_KEY is not configured
    """
    settings = get_settings()
    
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment variables")
    
    logger.info("Initializing Google Gemini embeddings via direct SDK")

    try:
        return GeminiSDKEmbeddings(
            api_key=settings.google_api_key,
            model_candidates=EMBEDDING_MODEL_CANDIDATES,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Google Gemini embeddings SDK wrapper: {str(e)}")
        raise


def embed_text(text: str) -> List[float]:
    """
    Embed a single text string using Google Gemini.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector as list of floats
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple text strings using Google Gemini.
    
    Args:
        texts: List of texts to embed
    
    Returns:
        List of embedding vectors
    """
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)
