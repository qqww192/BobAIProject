import streamlit as st
import os
import tempfile

# RAG & LangChain components
from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Google AI
from google import genai

# Update API key retrieval:
try:
    # Client now pulls the key directly from the environment
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except:
    st.error("API Key missing.")

# --- Initialization ---
GEMINI_MODEL = 'gemini-2.5-pro' 
PERSIST_DIR = "./default_db" 

GENERATION_CONFIG = {
    "temperature": 0.85,  # Higher temperature for conceptual leaps
    "top_p": 0.95,        # Broader range of token selection
    "top_k": 40,
    "max_output_tokens": 4096, # Ensure plenty of room for detailed advice
}

# Load the AI client
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except:
    st.error("API Key missing. Please check your .env file.")

# Load Persona Prompt
with open("persona_prompt.txt", "r") as f:
    SYSTEM_INSTRUCTION = f.read()

# Load Default Knowledge Base
# app.py (Near the top, where the function is defined)

# IMPORTANT: Ensure all necessary RAG imports are at the top of app.py:
# from langchain_community.document_loaders import TextLoader, PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings

SOURCE_DIR = "./source_docs"
PERSIST_DIR = "./default_db"

@st.cache_resource(show_spinner=False)
def load_default_db():
    embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 

    # --- Step 1: Try to Load Existing DB ---
    if os.path.exists(PERSIST_DIR):
        try:
            # Attempt to load the pre-existing DB
            st.success("Loading existing knowledge base...")
            return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings_model)
        except Exception as e:
            st.warning(f"Failed to load existing DB ({e}). Rebuilding knowledge base...")
            # If load fails, we proceed to rebuilding below.

    # --- Step 2: Rebuild DB if missing or failed to load ---
    if not os.path.exists(SOURCE_DIR) or not os.listdir(SOURCE_DIR):
        st.error("Cannot build knowledge base: 'source_docs' folder is missing or empty.")
        return None

    with st.spinner("Building Robert Palmer's foundational knowledge base..."):
        documents = []
        for filename in os.listdir(SOURCE_DIR):
            file_path = os.path.join(SOURCE_DIR, filename)
            if filename.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                except Exception as e:
                    st.warning(f"Warning: Failed to load PDF {filename}. Skipping. Error: {e}")
            elif filename.endswith(".txt"): 
                loader = TextLoader(file_path)
                documents.extend(loader.load())

        if not documents:
            st.error("No valid documents found in source_docs.")
            return None
            
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Create new vector store
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=embeddings_model,
            persist_directory=PERSIST_DIR 
        )
        vectordb.persist()
        st.success("Knowledge base built successfully.")
        return vectordb

VECTOR_STORE = load_default_db()

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Robert Palmer AI Advisor",
    layout="wide",
    initial_sidebar_state="collapsed" # Changed from auto/expanded
)

# 1. Header and Profile Section (Minimalist Design)
col1, col2 = st.columns([1, 4]) # 1:4 ratio for image vs text

with col1:
    # Display the profile image
    # IMPORTANT: Ensure profile.jpg is in the same directory as app.py
    try:
        st.image("profile.jpg", width=150)
    except FileNotFoundError:
        st.warning("Profile image not found. Ensure 'profile.jpg' is in the root directory.")

with col2:
    # Display the main title
    st.title("Robert Palmer: AI Cultural Policy Advisor")
    
    # Short, clear introduction
    st.markdown(
        """
        **Robert Palmer** is an experienced international cultural policy consultant, drawing on over four decades of strategic insight from roles including Director of Culture for the Council of Europe and advisory positions for UNESCO.
        
        His advice is **thoughtful, pragmatic, and non-judgmental**, focusing on sustainable processes, inclusive leadership, and the critical balance between passion and sustainability in the cultural sector.
        """
    )

st.markdown("---") # Visual separator

# 2. Sidebar for Grounding & Uploads (Kept functional)
# *** THIS REPLACES YOUR ORIGINAL SIDEBAR DEFINITION ***
with st.sidebar:
    st.header("Grounding & Context")
    st.markdown("Upload a document (e.g., a project proposal or policy paper) to ground Robert's advice specifically to your challenge.")
    uploaded_file = st.file_uploader("Upload PDF:", type=["pdf"]) # This must define uploaded_file
    st.markdown("---")
    st.caption("All advice is grounded in Robert Palmer's documented expertise.")
    
# 3. Chat Interface Start
st.header("Your Consultation") # New header for the chat area

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Optional: Initial welcome message from the AI to set the tone
    st.session_state.messages.append(
        {"role": "assistant", "content": "That's an interesting strategic question... How can I assist you with your cultural challenge today?"}
    )

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Continue in app.py

def get_rag_context(query, uploaded_file):
    """Retrieves relevant chunks from default and user-uploaded knowledge."""
    context_chunks = []

    # 1. Retrieve from Default Knowledge (Your friend's expertise)
    if VECTOR_STORE:
        default_results = VECTOR_STORE.similarity_search(query, k=3)
        context_chunks.extend([doc.page_content for doc in default_results])
    
    # 2. Handle User Uploaded File (Dynamic Grounding)
    user_context = ""
    if uploaded_file is not None:
        try:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Load and process the user file
            loader = PyPDFLoader(tmp_file_path)
            user_docs = loader.load()
            
            # Simple chunking for the user document
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            user_texts = text_splitter.split_documents(user_docs)
            
            # Simple search (or just use the first few chunks if the file is massive)
            user_context = "\n".join([t.page_content for t in user_texts[:5]]) # Limit context for efficiency
            
            context_chunks.append(f"\n--- USER SUPPLIED DOCUMENT (Grounding) ---\n{user_context}\n--- END USER DOCUMENT ---")
            
            # Clean up temp file
            os.remove(tmp_file_path)
            
        except Exception as e:
            st.warning(f"Could not process uploaded file: {e}")

    return "\n---\n".join(context_chunks)

# Continue in app.py
if prompt := st.chat_input("Ask for advice..."):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Robert Palmer is reflecting..."):
        # --- RAG Execution Start ---
        
        # Prepare temporary file path for uploaded doc (This logic needs to be inside the spinner block)
        # Note: We rely on the get_rag_context function handling the uploaded_file logic,
        # but we need to ensure the temporary file cleanup happens if needed. 
        # For simplicity, we assume get_rag_context handles temp file creation/cleanup internally
        # as structured in the previous comprehensive guide.

        # 1. Get Grounding Context
        grounding_context = get_rag_context(prompt, uploaded_file)
        
        # --- Prompt Construction (Strategies 1, 2, 3 integrated) ---

        # 2. Construct the Full Prompt for Gemini (Updated)
        full_prompt = f"""
{SYSTEM_INSTRUCTION} 

You MUST adhere strictly to the TONE, PRINCIPLES, and REASONING FRAMEWORK provided above.

--- KNOWLEDGE CONTEXT (Fixed Grounding & User Documents) ---
{grounding_context}
--- END KNOWLEDGE CONTEXT ---

You must integrate knowledge from the CONTEXT (both fixed and user-supplied) when relevant, using the Robert Palmer style and reasoning framework.

USER REQUEST: {prompt}

ROBERT PALMER'S RESPONSE: 
"""
        
        # 3. Call the Gemini API (Updated with config)
        try:
            # We need to transform the Python dict into the required GenerativeContentConfig object structure
            # when using the low-level SDK, though passing a simple dict often works in recent SDKs.
            # Assuming GENERATION_CONFIG is defined as a Python dictionary:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=full_prompt,
                config=GENERATION_CONFIG 
            )
            ai_response = response.text
        except Exception as e:
            ai_response = f"I apologize, I hit an internal error while seeking advice. Please check the API logs. Error: {e}"

    # 4. Display AI Response
    with st.chat_message("assistant"):
        st.markdown(ai_response)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})