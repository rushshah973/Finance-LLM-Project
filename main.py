import os
import langchain
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader,SeleniumURLLoader
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
import pandas as pd

os.environ['OPENAI_API_KEY'] = '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'
from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()

#UI using Streamlit
st.title("News Research Tool")
st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

file_path = "faiss_store_api.pkl"
main_placefolder = st.empty()
llm=OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.9,max_tokens=500)

process_url_click = st.sidebar.button("Process URLs")
#loaddata
if process_url_click:
        loader = SeleniumURLLoader(urls=urls)
        main_placefolder.text("Data loading ......Started")
        data=loader.load()
#splitdata
        text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ' '],
        chunk_size=1000
        )
        main_placefolder.text("Text Splitter ......Started")

        docs = text_splitter.split_documents(data )
# create embeddings
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placefolder.text("Embedding Vector......Started")

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)

query=main_placefolder.text_input("Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore=pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result= chain({"question":query},return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["answer"])
