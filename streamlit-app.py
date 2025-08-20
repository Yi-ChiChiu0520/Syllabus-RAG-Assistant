# Import required modules
import os
import time
import streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from ollama import Client

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Syllabus RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set paths
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"


# Define custom embedding function
class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode([f"passage: {text}" for text in input], show_progress_bar=False).tolist()


# Cache model loading
@st.cache_resource
def load_models():
    """Load and cache all models"""
    # ChromaDB setup
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_function = MyEmbeddingFunction()

    try:
        collection = chroma_client.get_collection(
            name="my-collection-original",
            embedding_function=embedding_function
        )
    except Exception as e:
        st.error(f"❌ Unable to load ChromaDB collection: {e}")
        st.info("💡 Make sure you have processed your syllabus documents first using the document processing script")
        st.stop()

    # Reranking model
    reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
    reranker_model.eval()

    return collection, reranker_tokenizer, reranker_model


@st.cache_data
def get_available_models():
    """Get list of available remote models"""
    try:
        base_url = "http://localhost:11434"  # Default Ollama URL
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models_data = response.json()
        return [model["name"] for model in models_data.get("models", [])]
    except Exception as e:
        st.warning(f"⚠️ Unable to get remote model list: {e}")
        return ["llama3.1:8b"]  # Default model


def setup_openai_client():
    """Setup OpenAI client"""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ Please set OPENAI_API_KEY in your .env file")
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        st.error("❌ Please install OpenAI library: pip install openai")
        return None


def setup_ollama_client():
    """Setup Ollama client"""
    try:
        base_url = "http://localhost:11434"  # Default Ollama URL
        return Client(host=base_url)
    except Exception as e:
        st.error(f"❌ Ollama client setup failed: {e}")
        st.info("💡 Make sure Ollama is running on your system")
        return None


def rewrite_query(user_query, model_type, client, model):
    """Rewrite query for better academic/syllabus search"""
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
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": rewriting_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.choices[0].message.content.strip()
        else:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": rewriting_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.get('message', {}).get('content', user_query).strip()
    except Exception as e:
        st.warning(f"⚠️ Query rewriting failed: {e}")
        return user_query


def generate_response(user_query, best_chunk, model_type, client, model):
    """Generate academic response based on syllabus content"""
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
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.choices[0].message.content
        else:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            return response.get('message', {}).get('content', '⚠️ Response format error')
    except Exception as e:
        return f"❌ Error generating response: {e}"


def process_query(user_query, collection, reranker_tokenizer, reranker_model, model_type, client, model):
    """Complete query processing pipeline"""
    results = {}

    # Step 1: Rewrite query
    with st.spinner("🔄 Optimizing your question for academic search..."):
        start_time = time.time()
        cleaned_query = rewrite_query(user_query, model_type, client, model)
        results['rewrite_time'] = time.time() - start_time
        results['cleaned_query'] = cleaned_query

    # Step 2: Retrieve documents
    with st.spinner("🔍 Searching through course syllabi..."):
        start_time = time.time()
        search_results = collection.query(
            query_texts=[cleaned_query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        results['retrieval_time'] = time.time() - start_time
        results['search_results'] = search_results

    # Step 3: Rerank results
    with st.spinner("🔢 Finding the most relevant course information..."):
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
                inputs = reranker_tokenizer(user_query, doc_text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    output = reranker_model(**inputs)
                    score = output.logits.item()

                if score > highest_score:
                    highest_score = score
                    best_chunk = doc_text
                    best_id = doc_id
                    best_metadata = metadata if metadata else {}

        results['rerank_time'] = time.time() - start_time
        results['best_chunk'] = best_chunk
        results['best_id'] = best_id
        results['best_score'] = highest_score
        results['best_metadata'] = best_metadata

    # Step 4: Generate response
    with st.spinner("🤖 Preparing your academic answer..."):
        start_time = time.time()
        response = generate_response(user_query, best_chunk, model_type, client, model)
        results['generation_time'] = time.time() - start_time
        results['response'] = response

    return results


# Main application
def main():
    st.title("📚 Syllabus RAG Assistant")
    st.markdown(
        "*Ask questions about your course syllabi and get instant, accurate answers from your academic materials*")

    # Load models
    with st.spinner("Loading academic models..."):
        collection, reranker_tokenizer, reranker_model = load_models()

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Model Configuration")

        # Model type selection
        model_type = st.radio(
            "Choose Model Type",
            ["OpenAI GPT-4o", "Local Ollama Model"],
            key="model_type"
        )

        if model_type == "OpenAI GPT-4o":
            client = setup_openai_client()
            model_name = "gpt-4o"
            model_type_key = "openai"
        else:
            client = setup_ollama_client()
            available_models = get_available_models()
            model_name = st.selectbox("Select Local Model", available_models)
            model_type_key = "ollama"

        if client is None:
            st.error("❌ Unable to setup selected model")
            st.stop()

        st.success(f"✅ Active Model: {model_name}")

        # Display collection info
        st.header("📊 Course Database Info")
        try:
            doc_count = collection.count()
            st.info(f"📂 Collection: syllabus-collection")
            st.info(f"📈 Documents: {doc_count}")

            if doc_count == 0:
                st.warning("⚠️ No syllabi found! Please add course documents to your database.")
        except Exception as e:
            st.error(f"Unable to get collection info: {e}")

        # Quick help
        with st.expander("💡 Example Questions"):
            st.markdown("""
            **Course Information:**
            - What are the learning objectives?
            - When are the exam dates?
            - What textbooks are required?

            **Assignments & Grading:**
            - What's the late submission policy?
            - How is the final grade calculated?
            - When is the final project due?

            **Policies:**
            - What's the attendance policy?
            - Are laptops allowed in class?
            - How do I contact the professor?
            """)

    # Main chat interface
    st.header("💬 Ask About Your Courses")

    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show detailed information (if available)
            if message["role"] == "assistant" and "details" in message:
                with st.expander("🔍 View Processing Details"):
                    details = message["details"]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🔄 Query Optimization", f"{details['rewrite_time']:.2f}s")
                        st.metric("🔍 Document Search", f"{details['retrieval_time']:.2f}s")

                    with col2:
                        st.metric("🔢 Relevance Ranking", f"{details['rerank_time']:.2f}s")
                        st.metric("🤖 Response Generation", f"{details['generation_time']:.2f}s")

                    total_time = (details['rewrite_time'] + details['retrieval_time'] +
                                  details['rerank_time'] + details['generation_time'])
                    st.metric("⏱️ Total Processing Time", f"{total_time:.2f}s")

                    st.markdown(f"**📘 Optimized Query:** {details['cleaned_query']}")
                    st.markdown(f"**🎯 Best Match ID:** {details['best_id']}")
                    st.markdown(f"**📊 Relevance Score:** {details['best_score']:.4f}")

                    # Show source file information
                    if 'best_metadata' in details and details['best_metadata']:
                        metadata = details['best_metadata']
                        if 'source' in metadata:
                            st.markdown(f"**📄 Source Document:** {metadata['source']}")

                    with st.expander("📄 Source Content Used"):
                        st.text_area(
                            "Syllabus Content",
                            value=details['best_chunk'],
                            height=200,
                            disabled=True,
                            key=f"source_{len(st.session_state.messages)}"
                        )

    # Chat input
    if prompt := st.chat_input("Ask me anything about your course syllabi..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query and generate response
        with st.chat_message("assistant"):
            try:
                # Process query
                results = process_query(
                    prompt, collection, reranker_tokenizer, reranker_model,
                    model_type_key, client, model_name
                )

                # Display response
                st.markdown(results['response'])

                # Immediately show detailed information
                with st.expander("🔍 View Processing Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🔄 Query Optimization", f"{results['rewrite_time']:.2f}s")
                        st.metric("🔍 Document Search", f"{results['retrieval_time']:.2f}s")

                    with col2:
                        st.metric("🔢 Relevance Ranking", f"{results['rerank_time']:.2f}s")
                        st.metric("🤖 Response Generation", f"{results['generation_time']:.2f}s")

                    total_time = (results['rewrite_time'] + results['retrieval_time'] +
                                  results['rerank_time'] + results['generation_time'])
                    st.metric("⏱️ Total Processing Time", f"{total_time:.2f}s")

                    st.markdown(f"**📘 Optimized Query:** {results['cleaned_query']}")
                    st.markdown(f"**🎯 Best Match ID:** {results['best_id']}")
                    st.markdown(f"**📊 Relevance Score:** {results['best_score']:.4f}")

                    # Show source file information
                    if 'best_metadata' in results and results['best_metadata']:
                        metadata = results['best_metadata']
                        if 'source' in metadata:
                            st.markdown(f"**📄 Source Document:** {metadata['source']}")

                    # Show complete source content
                    st.markdown("**📄 Complete Source Content Used:**")
                    st.text_area(
                        "Content",
                        value=results['best_chunk'],
                        height=200,
                        disabled=True,
                        key=f"chunk_{len(st.session_state.messages)}"
                    )

                # Save assistant response (including details)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": results['response'],
                    "details": results
                })

            except Exception as e:
                error_msg = f"❌ Error processing query: {e}"
                st.error(error_msg)
                # Also save error message to conversation history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # Clear conversation button
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    # Footer with usage tips
    st.markdown("---")
    st.markdown("""
    **📝 Tips for Better Results:**
    - Be specific about what you're looking for (dates, policies, requirements)
    - Ask about one topic at a time for clearer answers
    - Use academic terms when asking about course content
    - Check the source content in the details to verify information
    """)


if __name__ == "__main__":
    main()