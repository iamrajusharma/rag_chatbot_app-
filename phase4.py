import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Fixed imports for new LangChain versions
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

st.title("EVA Chatbot - Upload Your PDF")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=['pdf'],
        help="Upload a PDF document to ask questions about"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ {uploaded_file.name} uploaded!")
        
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing your PDF... This may take a moment."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Load and process PDF
                    loaders = [PyMuPDFLoader(tmp_file_path)]
                    
                    # Create vector index
                    index = VectorstoreIndexCreator(
                        embedding=HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        ),
                        text_splitter=RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                    ).from_loaders(loaders)
                    
                    # Store vectorstore in session state
                    st.session_state["vectorstore"] = index.vectorstore
                    st.session_state["pdf_processed"] = True
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    st.success("🎉 PDF processed successfully! You can now ask questions.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    # Clean up temp file on error
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
    
    # Show processing status
    if st.session_state["pdf_processed"]:
        st.success("✅ PDF Ready for Questions!")
    else:
        st.info("📤 Please upload and process a PDF first")

# Main chat interface
st.subheader("💬 Chat with Your PDF")

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg['role']).markdown(msg['context'])

# User input
prompt = st.chat_input("Ask me anything about your PDF document...")

if prompt:
    # Check if PDF is processed
    if not st.session_state["pdf_processed"] or st.session_state["vectorstore"] is None:
        st.warning("⚠️ Please upload and process a PDF first before asking questions!")
    else:
        # Show user message
        st.chat_message("user").markdown(prompt)
        st.session_state["messages"].append({"role": "user", "context": prompt})

        # Initialize Groq chat model
        groq_chat = ChatGroq(
            groq_api_key="your_real_key_here",
            model_name="llama-3.3-70b-versatile"
        )

        try:
            # Create RAG chain
            chain = RetrievalQA.from_chain_type(
                llm=groq_chat,
                chain_type='stuff',
                retriever=st.session_state["vectorstore"].as_retriever(
                    search_kwargs={"k": 3}
                ),
                return_source_documents=True
            )

            # Get response from RAG chain
            with st.spinner("Thinking..."):
                result = chain({"query": prompt})
                response = result["result"]
            
            # Show clean response
            st.chat_message("eva").markdown(response)
            st.session_state["messages"].append({"role": "eva", "context": response})
            
            # Optional: Show sources
            if "source_documents" in result and result["source_documents"]:
                with st.expander("📚 Source Content from PDF"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.write(f"**Source {i}:**")
                        st.write(doc.page_content[:300] + "...")
                        if hasattr(doc, 'metadata') and 'page' in doc.metadata:
                            st.write(f"*Page: {doc.metadata['page']}*")
                        st.write("---")
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            st.chat_message("eva").markdown(error_message)
            st.session_state["messages"].append({"role": "eva", "context": error_message})

# Instructions for users
with st.expander("ℹ️ How to Use"):
    st.markdown("""
    ### How to use EVA Chatbot:
    
    1. **Upload PDF**: Click on "Choose a PDF file" in the sidebar
    2. **Process**: Click "Process PDF" button and wait for completion
    3. **Ask Questions**: Type your questions in the chat input below
    4. **Get Answers**: EVA will answer based on your PDF content
    
    ### Tips:
    - 📄 Upload clear, text-based PDFs for best results
    - ❓ Ask specific questions about the content
    - 🔍 Check the source sections to see which parts of the PDF were used
    - 🔄 You can upload a new PDF anytime to replace the current one
    """)

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = []
    st.rerun()

# Reset PDF button
if st.sidebar.button("🔄 Reset PDF"):
    st.session_state["vectorstore"] = None
    st.session_state["pdf_processed"] = False
    st.session_state["messages"] = []
    st.rerun()