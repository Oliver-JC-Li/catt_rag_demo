from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st


openai_api_key = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(model='gpt-4o-mini', api_key=openai_api_key)

# Load the documents
loader = PyPDFLoader("../consent_forms/stop360/STOP360_Template.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
doc_chunks = text_splitter.split_documents(docs)
vector_db = FAISS.from_documents(doc_chunks, embedding=OpenAIEmbeddings())
retriever = vector_db.as_retriever()

# Contextualize question
context_q_system_prompt = ("""
Given a chat history and the latest user question which might
reference context in the chat history, formulate a standalone
question which can be understood without the chat history. Do
NOT answer the question, just reformulate it if needed and otherwise
return it as is.
""")

llm = ChatOpenAI(model='gpt-4o-mini')

context_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]

)
history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

# Question Answering
system_prompt = ("""
You are an assistant for question-answering tasks. Use the following
pieces of retrieved context to answer the question.
<context>
{context}
</context>
""")

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Manage chat history
chat_history = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Streamlit framework
vector_store = []
st.header("STOP360 conversational chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="I am a vertual study assistant, please as any question regarding STOP360 study")
    ]

if vector_store not in st.session_state:
    st.session_state.vector_store = retriever

user_input = st.chat_input("Ask you question over here")

if user_input is not None and user_input.strip() != "":
    response = conversational_rag_chain.invoke(
        {'input': user_input},
        config={
            "configurable": {"session_id": "stop360_1"}
        }
    )['answer']

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    else:
        with st.chat_message("Human"):
            st.write(message.content)