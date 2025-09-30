# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyMuPDFLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.chains import retrieval_qa





# st.title("EVA Chatbot")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # Display chat history
# for msg in st.session_state["messages"]:
#     st.chat_message(msg['role']).markdown(msg['context'])

# @st.cache_resource
# def get_vectorestore():
#     pdf_name=""
#     loaders=[PyMuPDFLoader("./Real.pdf")]
#     index=VectorstoreIndexCreator(
#         embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
#         text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

#     ).from_loaders(loaders)
#     return index.vectorstore




# # User input

# prompt = st.chat_input("Enter your prompt here")

# if prompt:
#     # Show user message
#     st.chat_message("user").markdown(prompt)
#     st.session_state["messages"].append({"role": "user", "context": prompt})

#     # Create prompt template (fixed: use 'question' instead of 'user')
#     groq_sys_prompt = ChatPromptTemplate.from_template(
#         "You are a knowledgeable assistant. Answer the question directly in one line without options.\n\nQuestion: {question}"
#     )

#     # Initialize Groq chat model
#     groq_chat = ChatGroq(
#         groq_api_key="your_real_key_here"",
#         model_name="llama-3.3-70b-versatile"
#     )

#     try:
#         vectorstore=get_vectorestore()
#         if vectorstore in None:
#             st.error("faild to load pdf")
            
#         chain=retrieval_qa.from_chain_type(
#           llm=groq_chat,
#           chain_type='stuff',
#           retriever=vectorstore.as_retriever(search_kwargs=({"k":3})),
#           return_source_document=True

#         )

#     # Create chain
#     # chain = groq_sys_prompt | groq_chat
#         result=chain({"question": prompt})
#         response=result["result"]

#     # try:
#     #     # Invoke with proper parameter name
#     #     response = chain.invoke({"question": prompt})
        
#         # Extract clean text from response
#         # if hasattr(response, 'content'):
#         #     response_text = response.content
#         # elif isinstance(response, dict) and "content" in response:
#         #     response_text = response["content"]
#         # else:
#         #     response_text = str(response)
        
#         # Show clean response
#         st.chat_message("eva").markdown(response)
#         st.session_state["messages"].append({"role": "eva", "context": response})
        
#     except Exception as e:
#         # error_message = f"Error: {str(e)}"
#         # st.chat_message("eva").markdown(error_message)
#         # st.session_state["messages"].append({"role": "eva", "context": error_message})
#         st.error(f"error:[str{e}]")

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# Fixed imports for new LangChain versions
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA  # Fixed import

st.title("EVA Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg['role']).markdown(msg['context'])

@st.cache_resource
def get_vectorstore():
    """Load and process PDF into vectorstore"""
    try:
        # Load PDF document
        loaders = [PyMuPDFLoader("./Real.pdf")]
        
        # Create vector index
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
        ).from_loaders(loaders)
        
        return index.vectorstore
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None

# User input
prompt = st.chat_input("Ask me about the PDF document...")

if prompt:
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state["messages"].append({"role": "user", "context": prompt})

    # Initialize Groq chat model
    groq_chat = ChatGroq(
        groq_api_key="your_real_key_here",
        model_name="llama-3.3-70b-versatile"
    )

    try:
        # Get vectorstore
        vectorstore = get_vectorstore()
        
        if vectorstore is None:  # Fixed syntax error
            st.error("Failed to load PDF")
        else:
            # Create RAG chain
            chain = RetrievalQA.from_chain_type(  # Fixed import name
                llm=groq_chat,
                chain_type='stuff',
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Fixed syntax
                return_source_documents=True  # Fixed parameter name
            )

            # Get response from RAG chain
            result = chain({"query": prompt})  # Changed from "question" to "query"
            response = result["result"]
            
            # Show clean response
            st.chat_message("eva").markdown(response)
            st.session_state["messages"].append({"role": "eva", "context": response})
            
            # Optional: Show sources
            if "source_documents" in result:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(result["source_documents"], 1):
                        st.write(f"**Source {i}:**")
                        st.write(doc.page_content[:200] + "...")
                        st.write("---")
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        st.chat_message("eva").markdown(error_message)
        st.session_state["messages"].append({"role": "eva", "context": error_message})