import streamlit as st
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
load_dotenv()

st.title("RAG Application built on Gemini Model")

#Se voce quiser utilizar um pdf em vez de site, tire o comentario das duas linhas de codigo abaixo e comente as outras que recebem as URLS
# loader = PyPDFLoader("")
# data = loader.load()


#Coloque aqui o seu Site
urls = ['']
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",temperature=0,max_tokens=None,timeout=None)

query = st.chat_input("Digite uma pergunta: ") 
#prompt = query

system_prompt = (
    "Você é um assistente para tarefas de resposta a perguntas."
    "Use as seguintes partes do contexto recuperado para responder "
    "a pergunta. Se você não sabe a resposta, diga que você não sabe"
    "Mantenha a resposta concisa."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": query})
    #print(response["answer"])

    st.write(response["answer"])


