import os
import glob
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path doesn't exist")
        
        logger.info(f"Loading files from {DATA_PATH}")

        # Get all PDF files in the directory
        pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found in the directory")
            return []

        documents = []
        successful_files = []
        failed_files = []

        # Load each PDF file individually to handle errors gracefully
        for pdf_file in pdf_files:
            try:
                logger.info(f"Attempting to load: {os.path.basename(pdf_file)}")
                
                # Load individual PDF
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                
                if docs:
                    documents.extend(docs)
                    successful_files.append(os.path.basename(pdf_file))
                    logger.info(f"✅ Successfully loaded: {os.path.basename(pdf_file)} ({len(docs)} pages)")
                else:
                    failed_files.append(f"{os.path.basename(pdf_file)} (empty)")
                    logger.warning(f"⚠️ File is empty: {os.path.basename(pdf_file)}")
                    
            except Exception as e:
                failed_files.append(f"{os.path.basename(pdf_file)} ({str(e)[:50]}...)")
                logger.warning(f"❌ Failed to load: {os.path.basename(pdf_file)} - Error: {str(e)}")
                continue

        # Log summary
        logger.info(f"📊 Loading Summary:")
        logger.info(f"   ✅ Successfully loaded: {len(successful_files)} files")
        logger.info(f"   ❌ Failed to load: {len(failed_files)} files")
        
        if successful_files:
            logger.info(f"   📄 Successful files: {', '.join(successful_files)}")
        
        if failed_files:
            logger.warning(f"   ⚠️ Failed files: {', '.join(failed_files)}")

        if not documents:
            logger.error("No documents were successfully loaded!")
            return []
        else:
            logger.info(f"🎉 Total documents loaded: {len(documents)} pages")

        return documents
    
    except Exception as e:
        error_message = CustomException("Failed to load PDFs", e)
        logger.error(str(error_message))
        return []

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents were found")
        
        logger.info(f"Splitting {len(documents)} documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        text_chunks = text_splitter.split_documents(documents)

        # Filter out empty chunks
        text_chunks = [chunk for chunk in text_chunks if chunk.page_content.strip()]

        logger.info(f"Generated {len(text_chunks)} text chunks")
        
        if not text_chunks:
            logger.error("No valid text chunks were generated!")
            return []
            
        return text_chunks
    
    except Exception as e:
        error_message = CustomException("Failed to generate chunks", e)
        logger.error(str(error_message))
        return []

def validate_pdf_file(file_path):
    """
    Validate if a PDF file can be read properly
    """
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return len(docs) > 0
    except Exception:
        return False

def get_pdf_file_info():
    """
    Get information about all PDF files in the data directory
    """
    if not os.path.exists(DATA_PATH):
        return {"error": "Data path doesn't exist"}
    
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    file_info = []
    
    for pdf_file in pdf_files:
        info = {
            "name": os.path.basename(pdf_file),
            "size": f"{os.path.getsize(pdf_file) / 1024:.1f} KB",
            "readable": validate_pdf_file(pdf_file)
        }
        file_info.append(info)
    
    return {
        "total_files": len(pdf_files),
        "files": file_info
    }