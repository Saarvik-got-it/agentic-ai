"""
Document ingestion pipeline.
Orchestrates loading, processing, and indexing documents into vector store.
Can be invoked separately for batch document updates.
"""

import time
from pathlib import Path
from utils.config import get_settings
from utils.logger import setup_logger
from utils.document_loader import load_documents, split_documents
from utils.vector_store import VectorStoreManager

logger = setup_logger(__name__)


def ingest_documents(document_folder: str = None, force_reindex: bool = False) -> bool:
    """
    Ingest documents from folder into vector store.
    Creates new index or loads existing one.
    
    Args:
        document_folder: Path to documents folder. Uses config if None.
        force_reindex: If True, recreate index even if it exists.
    
    Returns:
        True if successful, False otherwise
    """
    settings = get_settings()
    doc_folder = document_folder or settings.document_folder
    
    logger.info(f"Starting document ingestion from {doc_folder}")
    start_time = time.time()
    
    try:
        # Load documents
        logger.info("Step 1: Loading documents...")
        documents = load_documents(doc_folder)
        
        if not documents:
            logger.error("No documents found to ingest")
            return False
        
        # Split documents
        logger.info("Step 2: Splitting documents into chunks...")
        split_docs = split_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        logger.info(f"Created {len(split_docs)} chunks from {len(documents)} documents")
        
        # Initialize vector store
        logger.info("Step 3: Creating/updating vector store...")
        vector_manager = VectorStoreManager()
        
        # Force reindex if requested
        if force_reindex:
            logger.info("Force reindex requested - creating new vector store")
            vector_manager.create_vector_store(split_docs)
        else:
            logger.info("Loading or creating vector store...")
            vector_manager.get_or_create(split_docs)
        
        elapsed = time.time() - start_time
        logger.info(f"Document ingestion completed successfully in {elapsed:.2f}s")
        
        return True
    
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        return False


def reindex_documents(document_folder: str = None) -> bool:
    """
    Force reindexing of all documents (recreates vector store from scratch).
    
    Args:
        document_folder: Path to documents folder. Uses config if None.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Force reindexing requested")
    return ingest_documents(document_folder, force_reindex=True)
