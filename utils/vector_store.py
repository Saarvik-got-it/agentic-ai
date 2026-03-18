"""
Vector store abstraction layer for FAISS.
This abstraction allows easy migration to other vector DBs (MongoDB Atlas, Pinecone, etc.)
"""

from pathlib import Path
import time
from typing import List, Optional
# LangChain core abstractions moved under langchain_core in v0.1+.
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings
from utils.config import get_settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStoreManager:
    """
    Manages vector store creation, persistence, and retrieval.
    Abstracts FAISS implementation for easy swapping with alternative backends.
    """
    
    def __init__(self, vector_store_path: Optional[str] = None):
        """
        Initialize Vector Store Manager.
        
        Args:
            vector_store_path: Path to store/load FAISS index. If None, uses config path.
        """
        if vector_store_path is None:
            settings = get_settings()
            self.vector_store_path = settings.vector_store_path
        else:
            self.vector_store_path = vector_store_path
        
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
        self.faiss_index_path = Path(self.vector_store_path) / "index"
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vector store from documents.
        
        Args:
            documents: List of LangChain Document objects with content
        
        Returns:
            FAISS vector store instance
        
        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")
        
        logger.info(f"Creating vector store from {len(documents)} documents")
        start_time = time.time()
        
        try:
            embeddings = get_embeddings()
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            
            # Save to disk
            self._save_index()
            
            elapsed = time.time() - start_time
            logger.info(f"Vector store created and saved in {elapsed:.2f}s")
            
            return self.vector_store
        
        except Exception as e:
            logger.error(f"Failed to create vector store: {str(e)}")
            raise
    
    def load_vector_store(self) -> Optional[FAISS]:
        """
        Load existing FAISS vector store from disk.
        
        Returns:
            FAISS instance if exists, None otherwise
        """
        if not self.faiss_index_path.exists():
            logger.warning(f"Vector store not found at {self.faiss_index_path}")
            return None
        
        try:
            logger.info(f"Loading vector store from {self.faiss_index_path}")
            start_time = time.time()
            
            embeddings = get_embeddings()
            self.vector_store = FAISS.load_local(
                folder_path=str(self.faiss_index_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Vector store loaded in {elapsed:.2f}s")
            
            return self.vector_store
        
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return None
    
    def _save_index(self):
        """Save FAISS index to disk."""
        if self.vector_store:
            try:
                self.vector_store.save_local(str(self.faiss_index_path))
                logger.info(f"Vector store saved to {self.faiss_index_path}")
            except Exception as e:
                logger.error(f"Failed to save vector store: {str(e)}")
    
    def get_or_create(self, documents: List[Document]) -> FAISS:
        """
        Load existing vector store if available, otherwise create new one.
        
        Args:
            documents: Documents to use if creating new vector store
        
        Returns:
            FAISS vector store instance
        """
        # Try to load existing
        vector_store = self.load_vector_store()
        
        if vector_store:
            return vector_store
        
        # Create new if doesn't exist
        logger.info("Creating new vector store as no existing index found")
        return self.create_vector_store(documents)
    
    def retrieve_documents(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Retrieve top-k relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
        
        Returns:
            List of relevant Document objects
        
        Raises:
            RuntimeError: If vector store not loaded
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized. Call load_vector_store() first.")
        
        logger.info(f"Retrieving top-{k} documents for query: {query[:100]}")
        
        try:
            # Use similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(docs)} relevant documents")
            return docs
        
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            raise
    
    def get_retriever(self, k: int = 5):
        """
        Get a LangChain retriever from the vector store.
        Useful for integration with chains.
        
        Args:
            k: Number of documents to retrieve
        
        Returns:
            LangChain Retriever instance
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized")
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})


def initialize_vector_store(documents: List[Document]) -> VectorStoreManager:
    """
    Initialize vector store manager and load/create index.
    
    Args:
        documents: Documents to use if creating new store
    
    Returns:
        Initialized VectorStoreManager instance
    """
    manager = VectorStoreManager()
    manager.get_or_create(documents)
    return manager
