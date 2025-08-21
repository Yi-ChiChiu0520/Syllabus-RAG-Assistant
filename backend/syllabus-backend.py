from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import time
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client
import requests
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

# Global variables for model caching
chroma_client = None
embedding_function = None
reranker_tokenizer = None
reranker_model = None

# Configuration
CHROMA_PATH = "../chroma_db"
DATA_PATH = "../data"


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    collection_name: str
    model_type: str  # "openai" or "ollama"
    model_name: str


class QueryResponse(BaseModel):
    response: str
    rewrite_time: float
    retrieval_time: float
    rerank_time: float
    generation_time: float
    cleaned_query: str
    best_id: str
    best_score: float
    best_metadata: Dict[str, Any]
    best_chunk: str


class CollectionInfo(BaseModel):
    name: str
    document_count: int
    source_files: Dict[str, int]  # filename -> chunk count


class AvailableModels(BaseModel):
    ollama_models: List[str]
    openai_available: bool


# Custom embedding function
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode([f"passage: {text}" for text in input], show_progress_bar=False).tolist()


@asynccontextmanager
async def lifespan(app):
    global chroma_client, embedding_function, reranker_tokenizer, reranker_model
    print("🚀 Initializing models...")

    # Initialize resources
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_function = MyEmbeddingFunction()

    reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    reranker_model.eval()

    print("✅ Models initialized successfully")

    # Yield control to the app (serving happens between yield and cleanup)
    try:
        yield
    finally:
        # Optional: add any cleanup you need
        print("🧹 Shutting down… releasing resources")
        # e.g., del models if you want to force GC
        # global chroma_client, embedding_function, reranker_tokenizer, reranker_model
        # chroma_client = embedding_function = reranker_tokenizer = reranker_model = None


app = FastAPI(lifespan=lifespan)

# Add CORS middleware - THIS IS CRITICAL for frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In development - allow all origins
    # For production, specify your frontend URL:
    # allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

def get_openai_client():
    """Get OpenAI client if API key is available"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_ollama_client():
    """Get Ollama client"""
    try:
        base_url = "http://localhost:11434"
        return Client(host=base_url)
    except Exception:
        return None


def get_ollama_models():
    """Get available Ollama models"""
    try:
        base_url = "http://localhost:11434"
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models_data = response.json()
        return [model["name"] for model in models_data.get("models", [])]
    except Exception:
        return ["llama3.1:8b"]  # Default fallback


def summarize_collection_files(collection, sample_limit: int = 5000):
    """Get collection document count and source file summary"""
    try:
        data = collection.get(include=["metadatas"], limit=sample_limit)
        metadatas = data.get("metadatas", []) or []
        sources = []
        for m in metadatas:
            if isinstance(m, dict):
                src = m.get("source")
                if src:
                    # Extract filename from path
                    filename = os.path.basename(src)
                    sources.append(filename)
        return len(metadatas), Counter(sources)
    except Exception as e:
        print(f"Error summarizing collection: {e}")
        return 0, Counter()


def rewrite_query(user_query: str, model_type: str, model_name: str):
    """Rewrite query for better search"""
    rewriting_prompt = """You are an academic assistant helping to rewrite user queries for better syllabus and course content search. 
    Please rewrite the input question to be clear, error-free, and optimized for academic document retrieval. 
    Focus on academic terminology and course-related keywords while maintaining the original meaning.

    Examples:
    - "when is the test?" → "When are the exam dates and assessment schedule?"
    - "what do I need for class?" → "What are the required materials and textbooks for this course?"
    - "homework policy" → "What is the assignment submission policy and late work guidelines?"
    """

    try:
        if model_type == "openai":
            client = get_openai_client()
            if not client:
                return user_query

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": rewriting_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.choices[0].message.content.strip()
        else:
            client = get_ollama_client()
            if not client:
                return user_query

            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": rewriting_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.get('message', {}).get('content', user_query).strip()
    except Exception as e:
        print(f"Query rewriting failed: {e}")
        return user_query


def generate_response(user_query: str, best_chunk: str, model_type: str, model_name: str):
    """Generate response based on retrieved content"""
    system_prompt = f"""
    You are an Academic Assistant named "SyllabusBot" designed to help students understand their course materials and syllabi.

    Your guidelines:
    - Only answer based on the provided syllabus/course information
    - Be helpful, clear, and academically appropriate
    - If information is not in the provided content, say "I don't have that information in the available course materials"
    - Provide specific details like dates, requirements, policies when available
    - Use a friendly but professional academic tone
    - For assignment questions, include due dates and requirements if available
    - For policy questions, quote relevant sections when helpful

    Available course information:
    {best_chunk}
    """

    try:
        if model_type == "openai":
            client = get_openai_client()
            if not client:
                return "❌ OpenAI API key not configured"

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.choices[0].message.content
        else:
            client = get_ollama_client()
            if not client:
                return "❌ Ollama client not available"

            response = client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.get('message', {}).get('content', '⚠️ Response format error')
    except Exception as e:
        return f"❌ Error generating response: {e}"


# API Endpoints

@app.get("/")
async def root():
    return {"message": "Syllabus RAG API is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/collections", response_model=List[CollectionInfo])
async def get_collections():
    """Get all available collections with their info"""
    try:
        collections = chroma_client.list_collections()
        result = []

        for col in collections:
            collection = chroma_client.get_collection(
                name=col.name,
                embedding_function=embedding_function
            )
            doc_count, source_counter = summarize_collection_files(collection)

            result.append(CollectionInfo(
                name=col.name,
                document_count=doc_count,
                source_files=dict(source_counter)
            ))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collections: {e}")


@app.get("/models", response_model=AvailableModels)
async def get_available_models():
    """Get available language models"""
    ollama_models = get_ollama_models()
    openai_available = os.getenv("OPENAI_API_KEY") is not None

    return AvailableModels(
        ollama_models=ollama_models,
        openai_available=openai_available
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query and return response with details"""
    try:
        # Get collection
        collection = chroma_client.get_collection(
            name=request.collection_name,
            embedding_function=embedding_function
        )

        # Step 1: Rewrite query
        start_time = time.time()
        cleaned_query = rewrite_query(request.query, request.model_type, request.model_name)
        rewrite_time = time.time() - start_time

        # Step 2: Retrieve documents
        start_time = time.time()
        search_results = collection.query(
            query_texts=[cleaned_query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        retrieval_time = time.time() - start_time

        print("=== RAW SEARCH RESULTS ===")
        for i, (doc_id, doc_text, metadata) in enumerate(zip(
                search_results["ids"][0],
                search_results["documents"][0],
                search_results["metadatas"][0]
        )):
            source = metadata.get("source", "unknown") if metadata else "no metadata"
            print(f"Rank {i + 1}: {source} - Preview: {doc_text[:100]}")


        # Step 3: Rerank results
        start_time = time.time()
        highest_score = float("-inf")
        best_chunk = ""
        best_id = ""
        best_metadata = {}

        for doc_id_list, doc_list, metadata_list in zip(
                search_results["ids"],
                search_results["documents"],
                search_results["metadatas"]
        ):
            for doc_id, doc_text, metadata in zip(doc_id_list, doc_list, metadata_list):
                inputs = reranker_tokenizer(request.query, doc_text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    output = reranker_model(**inputs)
                    score = output.logits.item()

                if score > highest_score:
                    highest_score = score
                    best_chunk = doc_text
                    best_id = doc_id
                    best_metadata = metadata if metadata else {}

        rerank_time = time.time() - start_time

        # After reranking
        print(f"=== RERANKER SELECTED ===")
        print(f"Best source: {best_metadata.get('source', 'unknown')}")
        print(f"Best score: {highest_score}")

        # Step 4: Generate response
        start_time = time.time()
        response = generate_response(request.query, best_chunk, request.model_type, request.model_name)
        generation_time = time.time() - start_time

        return QueryResponse(
            response=response,
            rewrite_time=rewrite_time,
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            generation_time=generation_time,
            cleaned_query=cleaned_query,
            best_id=best_id,
            best_score=highest_score,
            best_metadata=best_metadata,
            best_chunk=best_chunk
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")


@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        # Check if collection exists first
        collections = chroma_client.list_collections()
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

        # Delete the collection
        chroma_client.delete_collection(name=collection_name)

        # Verify deletion
        updated_collections = chroma_client.list_collections()
        updated_names = [col.name for col in updated_collections]

        if collection_name in updated_names:
            raise HTTPException(status_code=500,
                                detail=f"Collection '{collection_name}' still exists after deletion attempt")

        return {
            "message": f"Collection '{collection_name}' deleted successfully",
            "deleted_collection": collection_name,
            "remaining_collections": updated_names
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@app.post("/upload/{collection_name}")
async def upload_documents(collection_name: str, files: List[UploadFile] = File(...)):
    """Upload and process documents to a collection"""
    try:
        from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, Docx2txtLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_data_path = os.path.join(temp_dir, "temp_data")
            os.makedirs(temp_data_path, exist_ok=True)

            # Save uploaded files
            saved_files = []
            file_info = []
            for uploaded_file in files:
                file_path = os.path.join(temp_data_path, uploaded_file.filename)
                with open(file_path, "wb") as f:
                    content = await uploaded_file.read()
                    f.write(content)
                saved_files.append(file_path)
                file_info.append({
                    "filename": uploaded_file.filename,
                    "size": len(content),
                    "type": uploaded_file.content_type
                })

            # Process documents
            raw_documents = []

            # Load PDF files
            pdf_files = [f for f in saved_files if f.lower().endswith('.pdf')]
            if pdf_files:
                pdf_loader = PyPDFDirectoryLoader(temp_data_path)
                pdf_documents = pdf_loader.load()
                raw_documents.extend(pdf_documents)

            # Load DOCX files
            docx_files = [f for f in saved_files if f.lower().endswith('.docx')]
            if docx_files:
                docx_loader = DirectoryLoader(
                    temp_data_path,
                    glob="**/*.docx",
                    loader_cls=Docx2txtLoader,
                    silent_errors=True
                )
                docx_documents = docx_loader.load()
                raw_documents.extend(docx_documents)

            # Load DOC files (basic support)
            doc_files = [f for f in saved_files if f.lower().endswith('.doc')]
            if doc_files:
                # Note: This requires LibreOffice or python-docx2txt for .doc files
                try:
                    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                    for doc_file in doc_files:
                        loader = UnstructuredWordDocumentLoader(doc_file)
                        doc_documents = loader.load()
                        raw_documents.extend(doc_documents)
                except Exception as e:
                    print(f"Could not process .doc files: {e}")

            if not raw_documents:
                raise HTTPException(status_code=400, detail="No documents could be processed")

            # Clean document content (remove newlines)
            for doc in raw_documents:
                doc.page_content = doc.page_content.replace("\n", "")

            # Create recursive character text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Maximum characters per chunk
                chunk_overlap=200,  # Overlapping characters between chunks
                separators=["\n\n", "\n", ".", "。", "！", "？", " ", ""]  # Separators (including Chinese punctuation)
            )

            chunks = text_splitter.split_documents(raw_documents)

            # Get or create collection
            collection = chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )

            # Get existing count
            existing_count = collection.count()

            # Prepare data for ChromaDB
            documents = []
            metadata = []
            ids = []

            for i, chunk in enumerate(chunks):
                text = chunk.page_content
                documents.append(text)

                # Create unique ID with timestamp
                unique_id = f"upload_{int(time.time())}_{i}"
                ids.append(unique_id)

                # Prepare metadata
                chunk_metadata = chunk.metadata.copy() if chunk.metadata else {}
                chunk_metadata.update({
                    'chunk_id': i,
                    'upload_timestamp': time.time(),
                    'chunk_size': len(text)
                })
                metadata.append(chunk_metadata)

            # Upload to collection
            collection.upsert(
                documents=documents,
                metadatas=metadata,
                ids=ids,
            )

            new_count = collection.count()
            added_count = new_count - existing_count

            return {
                "message": f"Successfully uploaded {len(files)} files to collection '{collection_name}'",
                "files_processed": file_info,
                "chunks_added": added_count,
                "total_chunks": new_count,
                "collection_name": collection_name
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)