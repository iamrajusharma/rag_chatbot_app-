# import streamlit as st
# import os

# from langchain_groq import ChatGroq
# # from langchain.output_parsers import StrOutputParser
# from langchain.prompts import ChatPromptTemplate




# # prompt = st.chat_input("Enter your prompt here")


# st.title("RAG Chatbot")

# if "messages" not in st.session_state:
#     st.session_state["messages"]=[]

# for  msg in st.session_state["messages"]:
#            st.chat_message(msg['role']).markdown(msg['context'])



# if prompt:
#     st.chat_message("user").markdown(prompt)

#     st.session_state["messages"].append({"role":"user","context":prompt})
#     groq_sys_prompt=ChatPromptTemplate.from_template("You are a knowledgeable assistant. Answer the question directly in one line without options.{user}")

#     model="openai/gpt-oss-20b"

#     groq_chat=ChatGroq(
#           groq_api_key="your_real_key_here"",
#           model_name=model

#     )
#     class MyStrParser:
#      def __call__(self, output):
#         return str(output)
#     chain=groq_sys_prompt | groq_chat |  MyStrParser ()

#     response=chain.invoke("prompt")


#     # # response="Hellow i am EVA"
#     # st.chat_message("eva").markdown(response)

#     # st.session_state.messages.append({"role":"eva","context":response})
#     # Suppose response = chain.invoke(prompt)
#     # response_text = response
#     # st.chat_message("eva").markdown(response_text)
#     # st.session_state["messages"].append({"role": "eva", "context": response_text})
#     if isinstance(response, dict) and "content" in response:
#         response_text = response["content"]  # sirf text
#     else:
#        response_text = str(response)  # agar string hi return hua ho

#     st.chat_message("eva").markdown(response_text)
#     st.session_state["messages"].append({"role": "eva", "context": response_text})

# prompt = st.chat_input("Enter your prompt here")

# if prompt:
#     st.chat_message("user").markdown(prompt)
#     st.session_state["messages"].append({"role": "user", "context": prompt})

#     groq_sys_prompt = ChatPromptTemplate.from_template(
#         "You are a knowledgeable assistant. Answer the question directly in one line without options.{user}"
#     )

#     groq_chat = ChatGroq(
#         groq_api_key=your_real_key_here",
#         model_name="openai/gpt-oss-20b"
#     )

#     class MyStrParser:
#         def __call__(self, output):
#             return str(output)

#     chain = groq_sys_prompt | groq_chat | MyStrParser()

#     # ✅ Pass actual user prompt, not string "prompt"
#     response = chain.invoke(prompt)

#     if isinstance(response, dict) and "content" in response:
#       response_text = response["content"]
#     else:
#        response_text = str(response)

#     st.chat_message("eva").markdown(response_text)
#     st.session_state["messages"].append({"role": "eva", "context": response_text})

# user input
# prompt = st.chat_input("Enter your prompt here")

# if prompt:
#     st.chat_message("user").markdown(prompt)
#     st.session_state["messages"].append({"role": "user", "context": prompt})

#     # Groq system prompt
#     groq_sys_prompt = ChatPromptTemplate.from_template(
#         "You are a knowledgeable assistant. Answer the question directly in one line without options.{question}"
#     )

#     # Groq chat model
#     groq_chat = ChatGroq(
#         groq_api_key=your_real_key_here"",
#         model_name="llama-3.3-70b-versatile"
#     )

#     # Parser to always convert to string
#     class MyStrParser:
#         def __call__(self, output):
#             # ✅ Extract content if dict, else convert to string
#             if isinstance(output, dict) and "content" in output:
#                 return str(output["content"])
#             return str(output)

#     # chain
#     chain = groq_sys_prompt | groq_chat | MyStrParser()

#     # invoke with actual user prompt
#     response_text = chain.invoke({"question":prompt})

#     # show response
#     st.chat_message("eva").markdown(response_text)
#     st.session_state["messages"].append({"role": "eva", "context": response_text})









#####################AI code




import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

st.title("EVA Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg['role']).markdown(msg['context'])

# User input
prompt = st.chat_input("Enter your prompt here")

if prompt:
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state["messages"].append({"role": "user", "context": prompt})

 
    groq_sys_prompt = ChatPromptTemplate.from_template(
        "You are a doctor. Answer the question directly in one line without options.\n\nQuestion: {question}"
    )

    # Initialize Groq chat model
    groq_chat = ChatGroq(
        groq_api_key="your_real_key_here",
        model_name="llama-3.3-70b-versatile"
    )

    # Create chain
    chain = groq_sys_prompt | groq_chat

    try:
       
        response = chain.invoke({"question": prompt})
        
       
        if hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, dict) and "content" in response:
            response_text = response["content"]
        else:
            response_text = str(response)
        
        
        st.chat_message("eva").markdown(response_text)
        st.session_state["messages"].append({"role": "eva", "context": response_text})
        
    except Exception as e:
        error_message = f"Error: {str(e)}"
        st.chat_message("eva").markdown(error_message)
        st.session_state["messages"].append({"role": "eva", "context": error_message})