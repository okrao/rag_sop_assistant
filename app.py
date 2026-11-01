import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

# 1. Load Environment & API
load_dotenv("../.env")
raw_key = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=raw_key.strip() if raw_key else None)

# 2. UI Configuration
st.set_page_config(page_title="RAG SOP Assistant", page_icon="📚")
st.title("Knowledge Retrieval Engine")
st.caption("Eliminate hold times by instantly querying complex policy documents.")

# 3. Sidebar: File Upload & Processing
with st.sidebar:
    st.header("Document Ingestion")
    # Added "pdf" to the allowed upload types
    uploaded_file = st.file_uploader("Upload an SOP (.txt, .pdf)", type=["txt", "pdf"])
    
    if uploaded_file and "vector_store" not in st.session_state:
        with st.spinner("Embedding document..."):
            
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Create a safe temporary path
            tmp_path = os.path.join(tempfile.gettempdir(), f"uploaded_doc_{uploaded_file.name}")
            
            # Explicitly open, write the bytes, and close the file
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Route to the correct LangChain loader
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)
                
            # RAG Pipeline: Load, Split, Embed, Store
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vector_store = Chroma.from_documents(chunks, embeddings)
            st.success(f"{uploaded_file.name} ingested and ready!")
# 4. Chat Memory Management
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. Chat Interface
if uploaded_file and "vector_store" in st.session_state:
    if rep_input := st.chat_input("Ask a question about this policy..."):
        
        with st.chat_message("user"):
            st.markdown(rep_input)
        st.session_state.messages.append({"role": "user", "content": rep_input})
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            results = st.session_state.vector_store.similarity_search(rep_input, k=2)
            context = "\n\n".join([doc.page_content for doc in results])
            
            system_prompt = f"""You are a strict compliance assistant for a contact center. 
            Answer the user's question using ONLY the following standard operating procedure text. 
            Do not use any outside knowledge.
            
            SOP CONTEXT:
            {context}"""
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": rep_input}
                ],
                model="openai/gpt-oss-120b",
                temperature=0.1
            )
            
            ai_response = chat_completion.choices[0].message.content
            message_placeholder.markdown(ai_response)
            
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
elif not uploaded_file:
    st.info("👈 Please upload an SOP document in the sidebar to begin.")