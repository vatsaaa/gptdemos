import os
import sys
import getopt
from dotenv import load_dotenv

## Setup DB
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

## Use DB
import pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain

## Use agents
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor, 
    load_chat_planner
)

def get_db():
    embeddings = OpenAIEmbeddings(model='gpt-3.5-turbo')
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    doc_db = Pinecone.from_existing_index(pinecone.list_indexes()[0], embeddings)

    return doc_db

def use_agents():
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print('Usage: python use_agents.py -h --help: Prints this help message')
    print('Usage: python use_agents.py -q --query: Some question to get the answer about')

    return

def process_args(args):
    query = None

    try:
        opts, args = getopt.getopt(args, "hq:",
                                   [
                                       "help",
                                       "query="
                                   ])
    except getopt.GetoptError as err:
        print("Error: " + str(err))
        use_agents()
        exit(-2)

    for o, a in opts:
        if o in ("-h", "--help"):
            use_agents()
            exit()
        elif o in ("-q", "--query"):
            query = a
        else:
            assert False, "Invalid option: " + o + "provided!"

    if query == None:
        use_agents()
        exit()

    return query

