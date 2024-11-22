from utils import *

'''
RetrievalQA is a wrapper around a specific prompt. Chain type “stuff“ uses a prompt assuming 
the whole query text fits into the context window. It uses the following prompt template:

Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}

{question}
'''

def check_db_reference():
    queries = [
        "What were the most important events for Google in 2022?",
        "Summarise the financials for Google in Q1 of 2022?"
    ]

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=doc_db.as_retriever())

    result = qa.run(queries[1])

    return result

if __name__ == '__main__':
    load_dotenv()

    doc_db = get_db()

    with get_openai_callback() as cb:
        response = check_db_reference()
    
    print(response, "\n\n", cb)


