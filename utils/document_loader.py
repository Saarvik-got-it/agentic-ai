"""
Document loading utilities for PDF and TXT files.
Abstracts LangChain document loaders and provides a unified interface.
"""

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# LangChain core abstractions moved under langchain_core in v0.1+.
from langchain_core.documents import Document
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_documents(document_folder: str) -> List[Document]:
    """
    Load all PDF and TXT documents from a specified folder.
    
    Args:
        document_folder: Path to folder containing documents
    
    Returns:
        List of LangChain Document objects
    
    Raises:
        FileNotFoundError: If document folder doesn't exist
        RuntimeError: If no documents found or loading fails
    """
    folder_path = Path(document_folder)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Document folder not found: {document_folder}")
    
    documents = []
    
    # Find all PDF and TXT files
    pdf_files = list(folder_path.glob("**/*.pdf"))
    txt_files = list(folder_path.glob("**/*.txt"))
    
    total_files = len(pdf_files) + len(txt_files)
    
    if total_files == 0:
        logger.warning(f"No PDF or TXT files found in {document_folder}")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files and {len(txt_files)} TXT files")
    
    # Load PDF files
    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading PDF: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            
            # Add metadata for tracking
            for doc in docs:
                doc.metadata["source"] = str(pdf_file)
                doc.metadata["file_type"] = "pdf"
            
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {pdf_file.name}")
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_file.name}: {str(e)}")
    
    # Load TXT files
    for txt_file in txt_files:
        try:
            logger.info(f"Loading TXT: {txt_file.name}")
            loader = TextLoader(str(txt_file), encoding="utf-8")
            docs = loader.load()
            
            # Add metadata for tracking
            for doc in docs:
                doc.metadata["source"] = str(txt_file)
                doc.metadata["file_type"] = "txt"
            
            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {txt_file.name}")
        except Exception as e:
            logger.error(f"Failed to load TXT {txt_file.name}: {str(e)}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk (in characters)
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of split Document objects
    """
    # Text splitters are provided by the dedicated langchain_text_splitters package.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    if not documents:
        logger.warning("No documents to split")
        return []
    
    logger.info(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = splitter.split_documents(documents)
    logger.info(f"Created {len(split_docs)} chunks from {len(documents)} documents")
    
    return split_docs
