"""
This is a RAG chatbot to answer research question regarding COVID-19 Anti-CD14 treatment trial using
part 1 and part 2 informed consent forms
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import streamlit as st

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Load the document directory
loader = PyPDFDirectoryLoader("./consent_forms/catt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
doc_chunks = text_splitter.split_documents(docs)

# Generate vector database and embeddings for document chunks
vector_db = FAISS.from_documents(doc_chunks, embedding=OpenAIEmbeddings())

# Define LLM and prompt to use
llm = ChatOpenAI(model='gpt-4o-mini')

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Define document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Define retriever chain
retriever = vector_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# streamlit framework
st.title("CaTT chatbot to answer research question")
input_text = st.text_input("Please feel free to ask any question regarding research study")

if input_text:
    st.write(retrieval_chain.invoke({'input': input_text}))


## Would feeding the whole document as one chunk provide better performance?
