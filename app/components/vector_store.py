from langchain_community.vectorstores import FAISS
import os
from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()

        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading existing vectorstore...")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            logger.warning("No vector store found..")
            return None

    except Exception as e:
        error_message = CustomException("Failed to load vectorstore", e)
        logger.error(str(error_message))
        return None

def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No chunks were found..")
        
        logger.info(f"Generating vectorstore from {len(text_chunks)} chunks")

        embedding_model = get_embedding_model()

        # Process in smaller batches to avoid memory issues
        batch_size = 50  # Process 50 chunks at a time
        
        if len(text_chunks) > batch_size:
            logger.info(f"Processing {len(text_chunks)} chunks in batches of {batch_size}")
            
            # Create initial database with first batch
            first_batch = text_chunks[:batch_size]
            logger.info(f"Creating initial vectorstore with {len(first_batch)} chunks...")
            db = FAISS.from_documents(first_batch, embedding_model)
            
            # Add remaining chunks in batches
            for i in range(batch_size, len(text_chunks), batch_size):
                batch_end = min(i + batch_size, len(text_chunks))
                current_batch = text_chunks[i:batch_end]
                
                logger.info(f"Adding batch {i//batch_size + 1}: chunks {i+1}-{batch_end}")
                
                # Create temporary FAISS for this batch
                temp_db = FAISS.from_documents(current_batch, embedding_model)
                
                # Merge with main database
                db.merge_from(temp_db)
        else:
            logger.info("Creating vectorstore with all chunks at once...")
            db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info("Saving vectorstore...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        
        db.save_local(DB_FAISS_PATH)

        logger.info("Vectorstore saved successfully...")

        return db
    
    except Exception as e:
        error_message = CustomException("Failed to create new vectorstore", e)
        logger.error(str(error_message))
        return None