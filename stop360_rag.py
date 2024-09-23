"""
This is a RAG chatbot to answer research question regarding antibiotics treatment for
pulmonary exacerbations in people with cystic fibrosis (CF)
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Load the document directory
docs = PyPDFLoader("./stop360/STOP360_Template.pdf").load()
doc_chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100).split_documents(docs)

# Generate vector database and embeddings for document chunks
vector_db = FAISS.from_documents(doc_chunks, embedding=OpenAIEmbeddings())

# Define LLM and prompt to use
llm = ChatOpenAI(model='gpt-4o-mini', api_key=openai_api_key)

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
st.title("STOP360 chatbot with template document to answer research question")
input_text = st.text_input("Please feel free to ask any question regarding research study")

if input_text:
    st.write(retrieval_chain.invoke({'input': input_text}))
