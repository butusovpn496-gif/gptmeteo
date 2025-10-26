import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

api_key = st.secrets["openai_api_key"]
llm = ChatOpenAI(model="gpt-5-nano", openai_api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langgraph.graph import START, StateGraph
# from typing_extensions import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("meteo_rag.pdf")
# pages = loader.load()

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Индексируем чанки
_ = vector_store.add_documents(documents=all_splits)



def generate_response(question):

    from langchain_core.prompts import ChatPromptTemplate

    retrieved_docs = vector_store.similarity_search(question)

    context = '\n'.join([doc.page_content for doc in retrieved_docs])

    prompt_template = ChatPromptTemplate([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
                "If you don't know the answer, just say that you don't know.\n\nContext:\n {context}"),
        ("user", "{question}")
    ])

    chain = prompt_template | llm

    response = chain.invoke({
        "question": question,
        "context": context,
    })
    # print(response.content)

    return response

result = ""

with st.form(key='qa_form', clear_on_submit=True, border=True):
    query_text = st.text_input(
    'Отправьте свой вопрос LLM:',
    placeholder='Здесь нужно написать вопрос',
    # disabled=not uploaded_file
)
    submitted = st.form_submit_button("Отправить")

    if submitted:
        with st.spinner('Calculating...'):
            # Генерируем ответ с помощью функции
            response = generate_response(query_text)
            result = response

# Отображаем результат, если он есть
if result:
    st.info(result.content)
