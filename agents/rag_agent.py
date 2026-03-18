"""
Main RAG Agent module.
Orchestrates document loading, embedding, retrieval, and response generation.
"""

import time
from typing import Optional
from utils.config import get_settings
from utils.logger import setup_logger
from utils.document_loader import load_documents, split_documents
from utils.vector_store import VectorStoreManager
from utils.llm import generate_with_fallback, create_rag_prompt

logger = setup_logger(__name__)


class RAGAgent:
    """
    Production-ready RAG Agent for document-based question answering.
    
    Features:
    - Loads and processes documents from local folder
    - Creates embeddings using Google Gemini
    - Stores vectors in FAISS with persistence
    - Retrieves relevant context for queries
    - Generates answers using fallback LLM mechanism
    - Comprehensive logging and error handling
    """
    
    def __init__(self, vector_store_manager: Optional[VectorStoreManager] = None):
        """
        Initialize RAG Agent.
        
        Args:
            vector_store_manager: Optional pre-initialized VectorStoreManager.
                                 If None, will initialize on first query.
        """
        self.settings = get_settings()
        self.vector_store_manager = vector_store_manager
        self.documents = None
        self.is_initialized = False
        logger.info("RAG Agent initialized")
    
    def initialize(self) -> bool:
        """
        Initialize RAG agent by loading documents and creating vector store.
        This is called automatically on first query if not already done.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            logger.info("RAG Agent already initialized")
            return True
        
        try:
            logger.info("Starting RAG Agent initialization...")
            start_time = time.time()
            
            # Step 1: Load documents
            logger.info(f"Loading documents from {self.settings.document_folder}")
            self.documents = load_documents(self.settings.document_folder)
            
            if not self.documents:
                logger.warning("No documents loaded. RAG Agent ready but no knowledge base.")
                self.is_initialized = True
                return True
            
            # Step 2: Split documents
            logger.info("Splitting documents into chunks...")
            split_docs = split_documents(
                self.documents,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap
            )
            
            # Step 3: Create/load vector store
            if not self.vector_store_manager:
                self.vector_store_manager = VectorStoreManager()
            
            self.vector_store_manager.get_or_create(split_docs)
            
            elapsed = time.time() - start_time
            logger.info(f"RAG Agent initialization completed in {elapsed:.2f}s")
            self.is_initialized = True
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {str(e)}")
            return False
    
    def query(self, user_query: str) -> str:
        """
        Process a user query and return an answer using RAG.
        
        Args:
            user_query: User's question/query
        
        Returns:
            Generated answer based on retrieved context
        
        Raises:
            ValueError: If user_query is empty
            RuntimeError: If initialization fails
        """
        if not user_query or not user_query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Processing query: {user_query[:100]}")
        start_time = time.time()
        
        try:
            # Auto-initialize if needed
            if not self.is_initialized:
                logger.info("Auto-initializing RAG Agent...")
                if not self.initialize():
                    raise RuntimeError("Failed to initialize RAG Agent")
            
            # If no documents loaded, return message
            if not self.documents:
                logger.warning("No documents in knowledge base")
                return "No documents found in the knowledge base. Please add documents to the data/documents folder."
            
            # Retrieve relevant documents
            logger.info(f"Retrieving top-{self.settings.top_k_chunks} relevant documents...")
            retrieved_docs = self.vector_store_manager.retrieve_documents(
                query=user_query,
                k=self.settings.top_k_chunks
            )
            
            if not retrieved_docs:
                logger.warning("No relevant documents retrieved")
                return "Not enough information provided in the documents."
            
            # Build context from retrieved documents
            context = "\n\n".join([
                f"[Document: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in retrieved_docs
            ])
            
            logger.info(f"Built context from {len(retrieved_docs)} documents")
            
            # Generate response using LLM with fallback
            logger.info("Generating response using LLM...")
            prompt = create_rag_prompt(context, user_query)
            response = generate_with_fallback(prompt)
            
            elapsed = time.time() - start_time
            logger.info(f"Query processed in {elapsed:.2f}s. Response length: {len(response)} chars")
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"


def rag_agent(query: str) -> str:
    """
    Convenience function for RAG query processing.
    Creates agent instance, initializes, and returns answer.
    
    Args:
        query: User's question
    
    Returns:
        Answer based on documents
    
    Note:
        This is the primary interface for integrating RAG agent into other systems
        (e.g., FastAPI backend, other agents).
    """
    try:
        agent = RAGAgent()
        # Auto-initialization happens on first query
        return agent.query(query)
    except Exception as e:
        logger.error(f"RAG agent error: {str(e)}")
        return f"RAG Agent Error: {str(e)}"
