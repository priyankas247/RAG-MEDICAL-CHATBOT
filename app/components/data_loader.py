import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

# Step 1: Ensure directory exists
VECTORSTORE_PATH = "vectorstore/db_faiss"
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

# Step 2: Load your PDFs
docs = []
data_path = "data"
for file in os.listdir(data_path):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(data_path, file))
        docs.extend(loader.load())

# Step 3: Split documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Step 4: Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Number of splits:", len(splits))
print("First split:", splits[0].page_content[:200] if splits else "No splits found")
test_vec = embeddings.embed_query("Hello world")
print("Embedding size:", len(test_vec))

# Step 5: Build vector store
db = FAISS.from_documents(splits, embeddings)

# Step 6: Save vector store
db.save_local(VECTORSTORE_PATH)

print(f"✅ FAISS vector store saved at: {os.path.abspath(VECTORSTORE_PATH)}")
