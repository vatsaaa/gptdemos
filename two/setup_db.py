from utils import *

def setup_documents_db():
    loader = DirectoryLoader('./Alphabet/', glob='**/*.pdf', show_progress=False)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=40)
    docs_split = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo')
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

    doc_db = Pinecone.from_documents(docs_split, embeddings, index_name='fwtest')
    return doc_db

def test_db_setup(doc_db, query):
    query = "What were the most important events for Google in 2022?"
    search_docs = doc_db.similarity_search(query)

    return search_docs

if __name__ == '__main__':
    load_dotenv()

    doc_db = setup_documents_db()

    with get_openai_callback() as cb:
        response = test_db_setup(doc_db)
    
    print(response, "\n\n", cb)
