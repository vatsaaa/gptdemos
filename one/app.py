import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from utils import *

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
 
def main():
    st.header("Querying PDFs in multiple languages 💬")

    # Upload PDF file/s
    pdfs = st.file_uploader(label=" ", type='pdf', accept_multiple_files=True)

    ## TODO: Remove names of the files that were uploaded

    # Read PDF file/s
    for pdf in pdfs:
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            # Get the text
            text = ""
            p_count = 0
            for page in pdf_reader.pages:
                p_count = p_count + 1
                text += page.extract_text()
    
            # Split the text into chunks to process
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text=text)

            ## Pickle them all to create embeddings
            store_name = pdf.name[:-4]
    
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"pickles/{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo')
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"pickles/{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

    # ChatGPT here
    ask = 0
    while True:
        ask = ask + 1
        # Accept user questions/query
        query = st.text_input("Ask questions about files uploaded:", key=ask)

        if query:
            docs = VectorStore.similarity_search(query=query, k=5)

            llm = ChatOpenAI(model_name='gpt-3.5-turbo')
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)

            st.write(response, "\n\n", cb)
        else:
            break
    
    # add_vertical_space(14)
    # st.write('&copy; [Ankur Vatsa](https://www.linkedin.com/in/ankurvatsa/)')
    footer()

if __name__ == '__main__':
    load_dotenv()
    setup_sidebar()
    main()
