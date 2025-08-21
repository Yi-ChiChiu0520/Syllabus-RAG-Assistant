import streamlit as st
import requests
import time
from typing import Dict, List, Any
import json
import re

# Configuration
API_BASE_URL = "http://localhost:8000"  # FastAPI backend URL

def wait_for_backend(max_wait_sec: int = 120, base_delay: float = 0.5) -> bool:
    """
    Ping /health with exponential backoff until backend is ready or timeout.
    """
    deadline = time.time() + max_wait_sec
    attempt = 0
    with st.status("🚀 Initializing backend (loading models)…", expanded=True) as status:
        while time.time() < deadline:
            attempt += 1
            try:
                r = requests.get(f"{API_BASE_URL}/health", timeout=5)
                if r.ok:
                    status.update(label="✅ Backend is ready.", state="complete")
                    return True
            except requests.exceptions.RequestException as e:
                pass  # keep retrying

            delay = min(base_delay * (2 ** (attempt - 1)), 5.0)  # cap per-wait at 5s
            status.write(f"Attempt {attempt}: backend not ready yet, retrying in {delay:.1f}s…")
            time.sleep(delay)

        status.update(label="❌ Backend not available within the wait window.", state="error")
        return False

# Set page configuration
st.set_page_config(
    page_title="Syllabus RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

if not wait_for_backend(max_wait_sec=180):  # give it up to 3 minutes if you like
    st.error("❌ Backend API is not available. Please ensure the FastAPI server is running on port 8000.")
    st.stop()

# API client functions
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_collections():
    """Get all collections from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/collections")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error connecting to API: {e}")
        return []


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_available_models():
    """Get available models from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error getting models: {e}")
        return {"ollama_models": ["llama3.1:8b"], "openai_available": False}


def query_api(query: str, collection_name: str, model_type: str, model_name: str):
    """Send query to the API and get response"""
    try:
        payload = {
            "query": query,
            "collection_name": collection_name,
            "model_type": model_type,
            "model_name": model_name
        }

        response = requests.post(f"{API_BASE_URL}/query", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error querying API: {e}")
        return None


def upload_documents(files, collection_name: str):
    """Upload files to the specified collection via API"""
    try:
        # Prepare files for upload
        files_data = []
        for uploaded_file in files:
            files_data.append(
                ("files", (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type))
            )

        response = requests.post(
            f"{API_BASE_URL}/upload/{collection_name}",
            files=files_data
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error uploading files: {e}")
        return None

def delete_collection_api(collection_name: str):
    """Delete a collection via API"""
    try:
        response = requests.delete(f"{API_BASE_URL}/collections/{collection_name}")
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error deleting collection: {e}")
        return False


def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def sanitize_collection_name_frontend(name: str) -> str:
    """Mirror backend sanitizer: keep letters, numbers, . _ - ; ensure len>=3"""
    clean = re.sub(r"[^a-zA-Z0-9._-]", "_", (name or "").strip()).strip("_")
    if len(clean) < 3:
        clean = f"{clean}_db"
    return clean

# Main application
def main():
    st.title("📚 Syllabus RAG Assistant")
    st.markdown(
        "*Ask questions about your course syllabi and get instant, accurate answers from your academic materials*")

    # Check API health
    if not check_api_health():
        st.error("❌ Backend API is not available. Please make sure the FastAPI server is running on port 8000.")
        st.code("python main.py", language="bash")
        st.stop()

    st.success("✅ Connected to backend API")

    # Initialize session state
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = "my-collection-original"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Model Configuration")

        # Get available models
        models_data = get_available_models()

        # Model type selection
        model_options = []
        if models_data["openai_available"]:
            model_options.append("OpenAI GPT-4o")
        if models_data["ollama_models"]:
            model_options.append("Local Ollama Model")

        if not model_options:
            st.error("❌ No models available")
            st.stop()

        model_type = st.radio("Choose Model Type", model_options, key="model_type")

        if model_type == "OpenAI GPT-4o":
            model_name = "gpt-4o"
            model_type_key = "openai"
        else:
            model_name = st.selectbox("Select Local Model", models_data["ollama_models"])
            model_type_key = "ollama"

        st.success(f"✅ Active Model: {model_name}")

        # Collection Management Section
        st.header("📂 Collection Management")

        # Get all available collections
        collections_data = get_collections()

        if not collections_data:
            st.warning("⚠️ No collections found in database")
            st.info("💡 Please run the document processing script to create collections")
        else:
            # Collection dropdown
            collection_names = [col["name"] for col in collections_data]

            try:
                default_index = collection_names.index(st.session_state.selected_collection)
            except ValueError:
                default_index = 0
                if collection_names:
                    st.session_state.selected_collection = collection_names[0]

            selected_collection = st.selectbox(
                "Select Active Collection",
                collection_names,
                index=default_index,
                key="collection_selector"
            )

            # Check if collection selection changed
            if selected_collection != st.session_state.selected_collection:
                st.session_state.selected_collection = selected_collection
                st.session_state.messages = []  # Clear chat history
                st.rerun()

            # Display current collection info
            st.markdown(f"**Active Collection:** `{st.session_state.selected_collection}`")

            # Find and display selected collection info
            selected_col_data = next(
                (col for col in collections_data if col["name"] == st.session_state.selected_collection),
                None
            )

            if selected_col_data:
                st.info(f"📈 Total Documents: {selected_col_data['document_count']}")

                if selected_col_data['source_files']:
                    with st.expander(f"📄 Files in Collection ({len(selected_col_data['source_files'])})"):
                        for i, (filename, count) in enumerate(sorted(selected_col_data['source_files'].items()), 1):
                            st.markdown(f"{i}. 📋 **{filename}** — {count} chunk(s)")
                else:
                    if selected_col_data['document_count'] > 0:
                        st.warning("⚠️ Documents found but no source files identified")
                    else:
                        st.warning("⚠️ This collection is empty")

                # Document Upload Section
                st.header("📁 Upload Documents")

                uploaded_files = st.file_uploader(
                    "📤 Drag and drop files or click to browse",
                    type=['pdf', 'docx', 'doc', 'txt', 'md'],
                    accept_multiple_files=True,
                    help="Upload syllabus files to add to the selected collection",
                    key="file_uploader"
                )

                # Choose destination: Existing or New
                upload_mode = st.radio(
                    "Upload to",
                    ["Existing collection", "Create new collection"],
                    horizontal=True,
                    key="upload_mode",
                )

                # Destination picker
                if upload_mode == "Existing collection":
                    target_collection = st.selectbox(
                        "🎯 Upload to collection:",
                        collection_names if collections_data else ["my-collection-original"],
                        index=collection_names.index(st.session_state.selected_collection)
                        if collections_data and st.session_state.selected_collection in collection_names else 0,
                        key="upload_target_collection_existing"
                    )
                else:
                    new_name_input = st.text_input(
                        "🆕 New collection name",
                        value="my-syllabus-collection",
                        help="Letters, numbers, dot, underscore and hyphen only (will be sanitized).",
                        key="upload_target_collection_new_input"
                    )
                    # Show live sanitized preview + collision warning
                    sanitized_preview = sanitize_collection_name_frontend(new_name_input)
                    st.caption(f"Sanitized name: `{sanitized_preview}`")
                    if sanitized_preview in collection_names:
                        st.warning("A collection with this name already exists. Pick a different name.")

                # Show selected files summary
                if uploaded_files:
                    st.markdown(f"**📋 Selected {len(uploaded_files)} file(s):**")
                    total_size = 0
                    for i, file in enumerate(uploaded_files, 1):
                        file_size_mb = file.size / (1024 * 1024)
                        total_size += file_size_mb
                        if file.type == "application/pdf":
                            icon = "📄"
                        elif file.type in [
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            "application/msword"
                        ]:
                            icon = "📝"
                        else:
                            icon = "📎"
                        st.markdown(f"{i}. {icon} **{file.name}** — {file_size_mb:.2f} MB")
                    st.info(f"💾 Total size: {total_size:.2f} MB")

                # Upload button
                if st.button("🚀 Upload Files", type="primary", use_container_width=True, key="upload_files_btn"):
                    if not uploaded_files:
                        st.error("❌ Please select at least one file.")
                    else:
                        # Resolve target collection per mode
                        if upload_mode == "Existing collection":
                            if not target_collection:
                                st.error("❌ Please select a target collection.")
                                st.stop()
                            destination = target_collection
                        else:
                            destination = sanitize_collection_name_frontend(new_name_input)
                            if not destination:
                                st.error("❌ Please enter a valid collection name.")
                                st.stop()
                            if destination in collection_names:
                                st.error("❌ Collection already exists. Choose a different name.")
                                st.stop()

                        with st.spinner(f"📤 Uploading {len(uploaded_files)} file(s) to '{destination}'..."):
                            result = upload_documents(uploaded_files, destination)

                        if result:
                            st.success("✅ Upload completed successfully!")

                            with st.expander("📊 Upload Results", expanded=True):
                                # The backend may return different keys; guard accordingly
                                st.write(f"**Collection:** {result.get('collection_name', destination)}")
                                st.write(f"**Files processed:** {len(result.get('files_processed', []))}")
                                st.write(f"**New chunks added:** {result.get('chunks_added', '—')}")
                                st.write(f"**Total chunks in collection:** {result.get('total_chunks', '—')}")

                                if result.get('files_processed'):
                                    st.markdown("**📁 Processed Files:**")
                                    for file_info in result['files_processed']:
                                        st.markdown(
                                            f"- ✅ {file_info.get('filename', '?')} ({file_info.get('size', '?')} bytes)")

                            # Refresh caches and UI
                            st.cache_data.clear()
                            time.sleep(1)
                            st.session_state.selected_collection = destination
                            st.session_state.messages = []
                            st.rerun()
                        else:
                            st.error("❌ Upload failed. Please check the backend logs.")


                else:
                    # Help text when no files selected
                    st.markdown("""
                    **📝 Supported file types:**
                    - 📄 **PDF** — Portable Document Format
                    - 📝 **DOCX** — Microsoft Word (modern)
                    - 📝 **DOC** — Microsoft Word (legacy)

                    **💡 Tips:**
                    - You can select multiple files at once
                    - Files will be processed and split into chunks
                    - Each chunk becomes searchable in the RAG system
                    """)

        # Collection Actions
        st.markdown("**Collection Actions**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Collections"):
                st.cache_data.clear()
                st.rerun()

            with col2:
                confirm_key = "confirm_delete"
                st.checkbox(f"⚠️ Confirm deletion of '{st.session_state.selected_collection}'", key=confirm_key)
                if st.button("⚠️ Delete Collection", help="Delete the currently selected collection"):
                    if not st.session_state.get(confirm_key):
                        st.warning("Please check the confirmation box first.")
                    else:
                        success = delete_collection_api(st.session_state.selected_collection)
                        if success:
                            st.success("✅ Collection deleted successfully")
                            # Reset to first available collection
                            remaining_collections = [c["name"] for c in collections_data if c["name"] != st.session_state.selected_collection]
                            if remaining_collections:
                                st.session_state.selected_collection = remaining_collections[0]
                            else:
                                st.session_state.selected_collection = "my-collection-original"
                            st.session_state.messages = []
                            st.cache_data.clear()
                            time.sleep(1)
                            st.rerun()

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
    st.caption(f"📂 Currently using: **{st.session_state.selected_collection}**")

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
                    if details['best_metadata'] and 'source' in details['best_metadata']:
                        st.markdown(f"**📄 Source Document:** {details['best_metadata']['source']}")

                    with st.expander("📄 Source Content Used"):
                        st.text_area(
                            "Syllabus Content",
                            value=details['best_chunk'],
                            height=200,
                            disabled=True,
                            key=f"source_{message.get('timestamp', time.time())}_{len(st.session_state.messages)}"
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
                # Show processing status
                with st.spinner("🤖 Processing your question..."):
                    # Query the API
                    results = query_api(
                        prompt,
                        st.session_state.selected_collection,
                        model_type_key,
                        model_name
                    )

                if results:
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
                        if results['best_metadata'] and 'source' in results['best_metadata']:
                            st.markdown(f"**📄 Source Document:** {results['best_metadata']['source']}")

                        # Show complete source content
                        st.markdown("**📄 Complete Source Content Used:**")
                        st.text_area(
                            "Content",
                            value=results['best_chunk'],
                            height=200,
                            disabled=True,
                            key=f"live_chunk_{time.time()}_{len(st.session_state.messages)}"
                        )

                    # Save assistant response (including details)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": results['response'],
                        "details": results,
                        "timestamp": time.time()
                    })
                else:
                    error_msg = "❌ Failed to get response from API"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "timestamp": time.time()
                    })

            except Exception as e:
                error_msg = f"❌ Error processing query: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": time.time()
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

    # API status footer
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("🔗 Connected to FastAPI backend")
    with col2:
        if st.button("🔄 Refresh API"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()