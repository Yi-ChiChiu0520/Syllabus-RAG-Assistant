# Important Note: Processing .doc files requires LibreOffice installation
# Windows: Download and install from https://www.libreoffice.org/download/, then add installation path to PATH
# Mac: brew install --cask libreoffice
# Linux: sudo apt-get install libreoffice
#
# Alternative solution: If LibreOffice installation has issues, manually convert .doc files to .docx format
# Conversion method: Open .doc files with Microsoft Word or Google Docs, save as .docx format
# Converted .docx files can be processed directly with Docx2txtLoader without LibreOffice

# Import required document loaders
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
# Import text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import ChromaDB related classes
from chromadb import Documents, EmbeddingFunction, Embeddings
# Import sentence transformer for generating embeddings
from sentence_transformers import SentenceTransformer
# Import tokenizer
from transformers import AutoTokenizer
import os
import chromadb
import ssl
import nltk

# Fix NLTK download SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)  # Sentence tokenizer
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)  # POS tagger
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"NLTK download warning (may still work normally): {e}")

# Set data path and ChromaDB path
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Create ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


# Custom embedding function class
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # Load BGE-M3 embedding model
        self.model = SentenceTransformer("BAAI/bge-m3")

    def __call__(self, input: Documents) -> Embeddings:
        # Convert text to embedding vectors, add "passage: " prefix to optimize retrieval
        return self.model.encode([f"passage: {text}" for text in input], show_progress_bar=False).tolist()


# Create embedding function instance
embedding_function = MyEmbeddingFunction()

# Get or create ChromaDB collection
collection = chroma_client.get_or_create_collection(
    name="my-collection-original",
    embedding_function=embedding_function
)

# Load documents in various formats
raw_documents = []

# Load PDF files
pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
pdf_documents = pdf_loader.load()
raw_documents.extend(pdf_documents)
print(f"Loaded {len(pdf_documents)} PDF documents")

# Load DOCX files (modern Word format)
doc_documents = []
try:
    docx_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        silent_errors=True  # Skip files that cannot be processed
    )
    docx_documents = docx_loader.load()
    doc_documents.extend(docx_documents)
    print(f"Loaded {len(docx_documents)} DOCX documents")
except Exception as e:
    print(f"Error occurred while loading DOCX files: {e}")


# Custom function to handle .doc files using multiple methods
# Note: This functionality requires LibreOffice to be installed on the system to work properly
def load_doc_files_manually():
    """Attempt to load .doc files using different methods

    Prerequisites:
    - Windows: Need to install LibreOffice and add to PATH environment variable
    - Mac: brew install --cask libreoffice
    - Linux: sudo apt-get install libreoffice

    Test after installation: Run 'soffice --version' in terminal to confirm availability

    Alternative solutions (if LibreOffice has issues):
    1. Manually convert .doc files to .docx format
    2. Using Microsoft Word: Open .doc file → Save As → Select .docx format
    3. Using Google Docs: Upload .doc → Download as .docx format
    4. Using online conversion tools: such as CloudConvert, OnlineConvert, etc.
    5. Converted .docx files can be processed directly by the Docx2txtLoader above
    """
    doc_files = []
    try:
        import glob
        # Find all .doc files
        doc_file_paths = glob.glob(os.path.join(DATA_PATH, "**/*.doc"), recursive=True)

        if not doc_file_paths:
            print("No .doc files found")
            return doc_files

        print(f"Found {len(doc_file_paths)} .doc files to process")

        for file_path in doc_file_paths:
            print(f"Attempting to load .doc file: {os.path.basename(file_path)}")
            success = False

            # Method 1: Try using UnstructuredWordDocumentLoader (requires LibreOffice)
            try:
                loader = UnstructuredWordDocumentLoader(
                    file_path,
                    mode="single"  # Use single mode to avoid complex parsing
                )
                documents = loader.load()
                if documents and documents[0].page_content.strip():
                    doc_files.extend(documents)
                    print(
                        f"✓ Successfully loaded .doc file using UnstructuredWordDocumentLoader: {os.path.basename(file_path)}")
                    success = True
                    continue
            except Exception as e:
                print(f"  - UnstructuredWordDocumentLoader failed: {str(e)[:100]}...")
                print("  Tip: If you see 'soffice command not found' error, please confirm LibreOffice is installed")
                print("  Or manually convert .doc files to .docx format, then re-run the program")

            if not success:
                print(f"✗ Unable to load .doc file: {os.path.basename(file_path)}")
                print("  Suggested solutions:")
                print("  1. Install LibreOffice and ensure 'soffice' command is available")
                print("  2. Or manually convert this file to .docx format")
                print("  3. After conversion, place .docx file in data folder, program will process automatically")

    except Exception as e:
        print(f"Error occurred while processing doc files: {e}")

    return doc_files

# Attempt to manually load .doc files
# Default using LibreOffice method
manual_doc_files = load_doc_files_manually()

# Windows users encountering LibreOffice issues can:
# 1. Comment out the line above
# 2. Uncomment the line below to use simplified version
# manual_doc_files = load_doc_files_manually_windows_fallback()

doc_documents.extend(manual_doc_files)

# Add all Word documents to raw documents list
raw_documents.extend(doc_documents)

print(f"Total documents loaded: {len(raw_documents)}")

# Clean document content (remove newlines)
for doc in raw_documents:
    doc.page_content = doc.page_content.replace("\n", "")

# Create recursive character text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum characters per chunk
    chunk_overlap=200,  # Overlapping characters between chunks
    separators=["\n\n", "\n", ".", "。", "！", "？", " ", ""]  # Separators (including Chinese punctuation)
)

# Split documents into chunks
chunks = text_splitter.split_documents(raw_documents)

# Load tokenizer corresponding to the embedding model
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

# Initialize storage lists
documents = []
metadata = []
ids = []
token_counts = []

# Process each document chunk
for i, chunk in enumerate(chunks):
    text = chunk.page_content
    # Calculate token count
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_counts.append(len(tokens))

    # Prepare data for ChromaDB storage
    documents.append(text)
    ids.append("ID" + str(i))  # Create unique ID
    metadata.append(chunk.metadata)  # Retain original metadata
    print(f"Chunk ID: {i} \n Text: {chunk}")

# Upload document chunks to ChromaDB collection
collection.upsert(
    documents=documents,
    metadatas=metadata,
    ids=ids,
)

# Test query functionality
data = collection.query(
    query_texts=["my query"],  # Query text
    include=["documents", "metadatas", "embeddings"],  # Included return data
)