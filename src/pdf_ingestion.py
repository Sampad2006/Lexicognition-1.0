from pathlib import Path
from typing import List, Optional
import logging

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# configuring python logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# This class processes PDF documents by:
#     1. Loading and extracting text with metadata
#     2. Chunking text into semantically meaningful segments
#     3. Preparing documents for vector database insertion
class PDFIngestionPipeline:
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200, #to keep overlap between 2 consecutive chunks
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter attempts to split on paragraph breaks first,
        # then sentences, then words, so custom order here
        if separators is None:
            separators = [
                "\n\n",  
                "\n",    
                ". ",    
                " ",     
                ""       
            ]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        logger.info(
            f"Initialized ingestion pipeline with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}" 
        )
    
    #load and extract text+metadata
    def load_pdf(self, file_path: str) -> List[Document]:
        # checking file path
        pdf_path = Path(file_path)
        
        if not pdf_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f" file not found at path: {file_path}")
        
        if not pdf_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a valid file: {file_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            logger.warning(f"File does not have .pdf extension: {file_path}")
            #if other extensions can carry pdfs
        
        try:
            logger.info(f"Loading PDF: {file_path}")
            
            # PyMuPDFLoader for text + metadata
            loader = PyMuPDFLoader(str(pdf_path)) #instance of the loader
            documents = loader.load()
            
            if not documents:
                raise ValueError(f"No content extracted from PDF (maybe extraction failure or file empty): {file_path}")
            
            logger.info(f"Successfully loaded {len(documents)} pages from PDF")
            
            #extra metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_path.name
                doc.metadata['file_path'] = str(pdf_path.absolute())
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading pdf: {str(e)}")
            raise Exception(f"Failed to load PDF '{file_path}': {str(e)}") from e
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.error("Empty documents list provided for chunking")
            raise ValueError("Cannot chunk empty document list")
        
        try:
            logger.info(f"Chunking {len(documents)} documents...")
            
            chunked_docs = self.text_splitter.split_documents(documents)
            
            for idx, doc in enumerate(chunked_docs):
                doc.metadata['chunk_index'] = idx
                doc.metadata['chunk_size'] = len(doc.page_content)
            
            logger.info(
                f"Created {len(chunked_docs)} chunks from {len(documents)} pages "
                f"(avg {len(chunked_docs) / len(documents):.1f} chunks per page)"
            )
            
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error during chunking: {str(e)}")
            raise Exception(f"Failed to chunk documents: {str(e)}") from e
    
    def ingest_pdf(self, file_path: str) -> List[Document]:
        logger.info(f"Starting PDF ingestion pipeline for: {file_path}")
        
        try:
            documents = self.load_pdf(file_path)

            chunked_documents = self.chunk_documents(documents)
            
            logger.info(
                f"Ingestion complete: {len(chunked_documents)} chunks ready for indexing"
            )
            
            return chunked_documents
            
        except Exception as e:
            logger.error(f"PDF ingestion pipeline failed: {str(e)}")
            raise


    def get_statistics(self, documents: List[Document]) -> dict:
        if not documents:
            return {
                'total_chunks': 0,
                'total_characters': 0,
                'avg_chunk_size': 0,
                'unique_pages': 0
            }
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        pages = set(doc.metadata.get('page', -1) for doc in documents)
        
        return {
            'total_chunks': len(documents),
            'total_characters': total_chars,
            'avg_chunk_size': total_chars / len(documents),
            'min_chunk_size': min(len(doc.page_content) for doc in documents),
            'max_chunk_size': max(len(doc.page_content) for doc in documents),
            'unique_pages': len(pages),
            'pages_with_content': sorted([p for p in pages if p >= 0])
        }
    


#testing usage
if __name__ == "__main__":
    pipeline = PDFIngestionPipeline(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    try:
        pdf_path = "attention.pdf" #placeholder 
        chunks = pipeline.ingest_pdf(pdf_path)

        stats = pipeline.get_statistics(chunks)
        print("\n=== Ingestion Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

            
        if chunks:
            print("\n=== Sample Chunk ===")
            sample = chunks[0]
            print(f"Content preview: {sample.page_content[:200]}...")
            print(f"Metadata: {sample.metadata}")
            
    except FileNotFoundError:
        print(f"Error: Please provide a valid PDF file path")
    except Exception as e:
        print(f"Error during ingestion: {str(e)}")
        