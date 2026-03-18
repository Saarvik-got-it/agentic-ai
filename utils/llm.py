"""
LLM wrapper with fallback mechanism across multiple Google Gemini models.
Implements automatic retry logic for rate limits, timeouts, and API errors.
"""

import time
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
# LangChain core message types moved under langchain_core in v0.1+.
from langchain_core.messages import HumanMessage, SystemMessage
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Model priority order (HIGH → LOW)
MODEL_PRIORITY = [
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash"
]

# Models to try (mapped to actual LangChain model names)
GEMINI_MODELS = {
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3-flash": "gemini-3-flash",
    "gemini-2.5-flash": "gemini-2.5-flash"
}


def get_llm_with_fallback(
    system_prompt: Optional[str] = None
) -> Optional[str]:
    """
    Create LLM instance with fallback mechanism.
    Attempts to initialize with first available model from priority list.
    
    Args:
        system_prompt: System prompt to guide LLM behavior
    
    Returns:
        Callable function that generates responses with fallback
    """
    settings = get_settings()
    
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY not set in environment variables")
    
    def generate_response(user_query: str) -> str:
        """
        Generate response using available Gemini models with automatic fallback.
        
        Args:
            user_query: User's query/prompt
        
        Returns:
            Generated response string
        """
        for model_name in MODEL_PRIORITY:
            llm_model = GEMINI_MODELS.get(model_name)
            if not llm_model:
                logger.warning(f"Model {model_name} not found in GEMINI_MODELS mapping")
                continue
            
            try:
                logger.info(f"Attempting to generate response with model: {model_name}")
                
                # Initialize LLM
                llm = ChatGoogleGenerativeAI(
                    model=llm_model,
                    google_api_key=settings.google_api_key,
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.llm_max_tokens,
                    timeout=30
                )
                
                # Build message list
                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=user_query))
                
                # Generate response
                response = llm.invoke(messages)
                logger.info(f"Successfully generated response using {model_name}")
                
                return response.content
            
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for rate limit or API errors
                if any(phrase in error_str for phrase in 
                       ["rate limit", "quota", "429", "503", "timeout", "temporarily unavailable"]):
                    logger.warning(f"Model {model_name} failed (rate limit/timeout): {str(e)}")
                    time.sleep(1)  # Brief delay before trying next model
                    continue
                
                # Check for other API errors
                elif any(phrase in error_str for phrase in 
                        ["api", "error", "failed", "connection"]):
                    logger.warning(f"Model {model_name} failed (API error): {str(e)}")
                    continue
                
                else:
                    logger.error(f"Unexpected error with {model_name}: {str(e)}")
                    continue
        
        # All models failed
        error_msg = "Unable to generate response. All Gemini models failed. Please try again later."
        logger.error(error_msg)
        return error_msg
    
    return generate_response


def generate_with_fallback(
    prompt: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Convenience function to generate response with fallback mechanism.
    
    Args:
        prompt: User query/prompt
        system_prompt: Optional system prompt for context
    
    Returns:
        Generated response or error message
    """
    logger.info("Starting response generation with fallback mechanism")
    
    generator = get_llm_with_fallback(system_prompt)
    response = generator(prompt)
    
    return response


def create_rag_prompt(
    context: str,
    query: str
) -> str:
    """
    Create a formatted RAG prompt with context and query.
    
    Args:
        context: Retrieved context from vector store
        query: User's original query
    
    Returns:
        Formatted prompt string
    """
    prompt = f"""Based on the following context, answer the user's query concisely and factually.
Do NOT hallucinate or make up information. If the answer cannot be found in the context, respond with "Not enough information provided in the documents."

Context:
{context}

Query: {query}

Answer:"""
    
    return prompt
