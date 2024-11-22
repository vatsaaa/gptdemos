import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from utils import *

ask = 0

def setup_sidebar():
    ## Setup the Sidebar
    with st.sidebar:
        st.markdown('''
            Powered by [OpenAI](https://platform.openai.com/docs/models) LLM models
            '''
        )
        add_vertical_space(1)
        st.write('&copy; [Ankur Vatsa](https://www.linkedin.com/in/ankurvatsa/)')

def footer():
    footer = """
        <style>
            footer {
                visibility: hidden;
            }
        </style>
    """
    st.markdown(footer, unsafe_allow_html=True) 
 
def get_text(pdfs):
    text = ""
    for pdf in pdfs:
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            # Get the text
            p_count = 0
            for page in pdf_reader.pages:
                p_count = p_count + 1
                text += page.extract_text()
    
    return text
    
def main():
    setup_sidebar()
    st.header("Querying PDFs in multiple languages 💬")

    pdfs = st.file_uploader(label=" ", type='pdf', accept_multiple_files=True)
    text = get_text(pdfs) if pdfs else ""

    if text:
        # Split the text into chunks to process
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
        chunks = text_splitter.split_text(text=text)

        store_name = "alphabet_earnings_release"
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"pickles/{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo')
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"pickles/{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # The model is as much creative as it is deterministic
        if VectorStore:
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
            cql = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
            # llm = AzureOpenAI(deployment_name="td2", model_name="text-davinci-002")
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            # chain = load_qa_chain(llm=llm, chain_type="stuff")
            chain = ConversationalRetrievalChain.from_llm(
                        llm=llm, 
                        memory=memory, 
                        retriever=VectorStore.as_retriever(), 
                        chain_type="stuff", 
                        condense_question_llm=cql
                    )

        chat_history = []
        while True:
            prev_ask = ask
            # Accept user questions/query
            query = st.text_input("Ask questions about files uploaded:", key=ask)
            if query:
                docs = VectorStore.similarity_search(query=query, k=5)
                with get_openai_callback() as cb:
                    # response = chain.run(input_docs=docs, question=query)
                    response = chain({"question": query, "chat_history": chat_history})

                chat_history = [(query, response["answer"])]
                st.write(response["answer"], "\n\n", cb)

            ask = ask + 1
    
    footer()

if __name__ == '__main__':
    load_dotenv()
    main()
