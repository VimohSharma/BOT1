from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseChatMessageHistory
import os
import streamlit.components.v1 as components
import os

components.html("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap" rel="stylesheet">
    <div style="font-family: 'Poppins', sans-serif; font-size: 24px; color="white">
        <h1>Conversational Bot With Given Context</h1>
        <p>This web-application acts as a PDF Question-Answer Chatbot, It can take multiple pdfs for context and it will try it's best to answer your queries</p>
        <p style="font-size: 0.8em; font-weight: bold; padding: 10px; color: white; background: linear-gradient(to right, #ff8a00, #e52e71); border-radius: 8px; text-align: center;">
    ⚠️ Open in wide mode
</p>
    </div>
""", height=400)

from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]


#groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

api_key=st.text_input("Enter your Groq API Key:",type="password")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store= None
if api_key:
    model = ChatGroq(model="Llama3-8b-8192",groq_api_key=api_key)

    session_id = st.text_input("Session ID",value="Dfault-Sesh")
    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_files=st.file_uploader("Drop your PDF File here",type="pdf",accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temp_pdf=f"temp.pdf"
            with open(temp_pdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
            
            loader=PyPDFLoader(temp_pdf) ## here we fill and initialise the loader class with the pdf or file 
            docs=loader.load() ## here we work and process the characters or content of the pdf file present in the loader to the docs which happens when we do .load()
            documents.extend(docs) ## here we append the pdf file history with the current processed data

        ## Splitting and creating embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        if vector_store is None:
            vector_store = FAISS.from_documents(splits,embeddings)
        else:
            vector_store.add_documents(splits)  
        retriever = vector_store.as_retriever()

        ## Creating new prompt with context
        contextualized_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history,"
        "formulate a standalone question which can be understood"
        "without the chat history. Don not answer the question"
        "just reformulate it if needed and otherwise return it as is"
    )
        contextualized_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualized_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

        history_aware_retriever=create_history_aware_retriever(model,retriever,contextualized_prompt)

        prompt=(
            "Answer the questions based on the provided context"
            "only. Please provide the most accurate response based"   
            "on the question."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system" , prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain=create_stuff_documents_chain(model,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        with_msg_history = RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key="input",history_messages_key="chat_history",output_messages_key="answer")

        user_input = st.text_input("Enter your query from the research paper")

        if user_input:
            session_history=get_session_history(session_id)
            response=with_msg_history.invoke(
                {"input":user_input},
                    config={
                "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter a valid Groq API Key")
