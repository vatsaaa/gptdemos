import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
 
# Setup the Sidebar
with st.sidebar:
    st.markdown('''
    ## About
    Multi-lingual pdf querying chatbot:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/) and
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('&copy; [Ankur Vatsa](https://www.linkedin.com/in/ankurvatsa/)')
 
# Fetch keys for OPENAI and Embeddings DB
load_dotenv() 
 
def main():
    st.header("Query PDF Documents in your language 💬")

    # Upload PDF file/s
    pdfs = st.file_uploader(" ", type='pdf', accept_multiple_files=True)
 
    # Read PDF file/s
    for pdf in pdfs:
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            # Get the text
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
    
            # Split the text into chunks to process
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text=text)

            ## Create embeddings
            store_name = pdf.name[:-4]
            st.write(f'{store_name}')
    
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo')
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

    # Accept user questions/query
    query = st.text_input("Ask questions about files uploaded:", key=1)
 
    if query:
        docs = VectorStore.similarity_search(query=query, k=5)

        llm = OpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)

if __name__ == '__main__':
    main()