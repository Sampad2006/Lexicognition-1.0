import os

import shutil   
import hashlib
import json
from pathlib import Path
from typing import List, Optional
import logging
import tempfile
import uuid

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    
    def __init__(
        self,
        session_id: str,
        persist_directory: str = "./persistent_storage/chroma_db",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "pdf_documents"
    ):
        # if persist_directory is None:
        #     # This ensures a unique, writable folder for every run
        #     persist_directory = os.path.join(tempfile.gettempdir(), "chroma_db")

        # self.persist_directory = os.path.abspath(persist_directory)
        # self.collection_name = collection_name
        
        # # explicitly creating the database directory structure for st cloud search issue
        # os.makedirs(self.persist_directory, exist_ok=True) #
        
        # logger.info(f"Initializing VectorStoreManager with model: {embedding_model_name}")
        
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # cache_dir = os.path.join(current_dir, "..", "persistent_storage", "model_cache")
        # os.makedirs(cache_dir, exist_ok=True)
        # Create a unique path in /tmp for this specific session
        self.persist_directory = os.path.join(tempfile.gettempdir(), f"chroma_{session_id}")
        self.collection_name = collection_name
        
        # Ensure the isolated directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Use a separate cache for models to avoid permission issues in the source tree
        cache_dir = os.path.join(tempfile.gettempdir(), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        #initially cpu
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            cache_folder=cache_dir,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Embedding model loaded. Cache location: {cache_dir}")
    
    def _get_source_info_path(self) -> Path:
        return Path(self.persist_directory) / "source_info.json"

    def _calculate_file_hash(self, filepath: str) -> str: #SHA-256 hashing haha
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()

    
    def database_exists(self) -> bool: #smart move ik
        db_path = Path(self.persist_directory)
        
        if not db_path.exists():
            return False
        
        if not any(db_path.iterdir()):
            return False
        
        logger.info(f"Existing database found at: {self.persist_directory}")
        return True
    
    def is_source_current(self, pdf_path: str) -> bool:
        source_info_path = self._get_source_info_path()
        if not source_info_path.exists():
            logger.info("Source info file not found. Assuming database is stale .So loads of work.")
            return False

        try:
            with open(source_info_path, "r") as f:
                stored_info = json.load(f)
            
            current_hash = self._calculate_file_hash(pdf_path)
            
            if stored_info.get("hash") == current_hash:
                logger.info("Source PDF matches stored hash. Database is current.")
                return True
            else:
                logger.warning("Source PDF hash mismatch. Database is stale.")
                return False
        except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
            logger.error(f"Error checking source info: {e}. Assuming database is stale.")
            return False
    
    def clear_database(self) -> None:
        db_path = Path(self.persist_directory)
        
        if db_path.exists():
            logger.warning(f"Clearing existing database at: {self.persist_directory}")
            shutil.rmtree(db_path)
            logger.info("Database cleared successfully")
        else:
            logger.info("No existing database to clear")
    
    def create_vector_store(
        self,
        chunks: List[Document],
        source_pdf_path: str 
    ) -> VectorStoreRetriever:
        if not chunks:
            logger.error("Empty chunks list provided")
            raise ValueError("Cannot create vector store from empty document list")
        
        self.clear_database()
        
        os.makedirs(self.persist_directory, exist_ok=True) #reinitializing for st cloud environment 

        logger.info(f"Creating new vector store from {len(chunks)} document chunks...")
        
        try:
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            source_hash = self._calculate_file_hash(source_pdf_path)
            source_info_path = self._get_source_info_path()
            with open(source_info_path, "w") as f:
                json.dump({
                    "source": os.path.basename(source_pdf_path), 
                    "hash": source_hash
                }, f)
            
            logger.info(f"Database persisted and fingerprint saved: {source_hash}")
            return self._create_retriever(vectorstore)
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def _create_retriever(
        self,
        vectorstore: Chroma,
        k: int = 3,
        search_type: str = "similarity"
    ) -> VectorStoreRetriever:
        """
        Create a configured retriever from a vector store.
        
        Args:
            vectorstore: ChromaDB vector store instance
            k: Number of top results to return (default: 3)
                - More results = better coverage but more noise
                - 3-5 is a good balance for most RAG applications
            search_type: Type of search to perform
                - "similarity": Pure cosine similarity (default)
                - "mmr": Maximum Marginal Relevance (diverse results)
        
        Returns:
            Configured VectorStoreRetriever ready for querying
        """
        logger.info(f"Creating retriever with k={k}, search_type={search_type}")
        
        retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        return retriever
    
    def load_existing_store(self, k: int = 3) -> VectorStoreRetriever:
        """
        Load an existing ChromaDB database and return a retriever.
        
        Use this when you want to query an existing database without
        adding new documents.
        
        Args:
            k: Number of results to return per query
        
        Returns:
            VectorStoreRetriever for the existing database
        
        Raises:
            FileNotFoundError: If database doesn't exist
            Exception: For loading errors
        """
        if not self.database_exists():
            raise FileNotFoundError(
                f"No database found at {self.persist_directory}. "
                "Please create one first using create_vector_store()."
            )
        
        logger.info(f"Loading existing vector store from: {self.persist_directory}")
        
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            
            logger.info("Vector store loaded successfully")
            return self._create_retriever(vectorstore, k=k)
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise Exception(f"Failed to load vector store: {str(e)}") from e
    
    def get_database_stats(self) -> dict:
        """
        Get statistics about the current database.
        
        Returns:
            Dictionary with database information
        """
        if not self.database_exists():
            return {
                'exists': False,
                'path': self.persist_directory,
                'collection_name': self.collection_name
            }
        
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            
            # Get collection info
            collection = vectorstore._collection
            
            return {
                'exists': True,
                'path': self.persist_directory,
                'collection_name': self.collection_name,
                'document_count': collection.count(),
                'embedding_dimension': len(self.embedding_model.embed_query("test"))
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {
                'exists': True,
                'path': self.persist_directory,
                'error': str(e)
            }


def create_vector_store(
    chunks: List[Document],
    source_pdf_path: str, # Added parameter
    persist_directory: str = "./persistent_storage/chroma_db",
    k: int = 3
) -> VectorStoreRetriever:
    manager = VectorStoreManager(persist_directory=persist_directory)
    return manager.create_vector_store(chunks, source_pdf_path=source_pdf_path)


# Example usage and testing
if __name__ == "__main__":
    
    
    print("=== 2nd part: Vector Store Creation ===\n")
    
    manager = VectorStoreManager(persist_directory="./chroma_db")
    
    stats = manager.get_database_stats()
    print("Database Status:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("part 2 module ready. Import and use create_vector_store() function.")