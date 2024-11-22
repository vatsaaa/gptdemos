'''
"agents", based on user input, decide the tools to utilize from a suite they 
have access to, with two main types being "Action Agents" that take actions 
one step at a time, and "Plan-and-Execute Agents" that decide a plan of actions 
first and then execute them one at a time. 
e.g. Give ChatGPT an access to Wikipedia and use the “ZERO_SHOT_REACT_DESCRIPTION“ flag, 
the LLM is able to understand how to use Wikipedia just based on the tool description.
'''

from utils import *

def setup_qa(doc_db, llm):
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )

    return qa

def setup_tools(qa, llm):
    ## Setup pdf search tool
    tool_name = """
    PDF_Search
    """

    tool_description = """
    PDF_Search tool is useful to answer questions about earnings of Alphabet
    in 2021, 2022 and 2023. Input may be a partial or fully formed question.
    """

    pdf_search = Tool(
        name=tool_name,
        func=qa.run,
        description=tool_description,
    )

    ## Tools pipeline
    # tools.append(pdf_search)

    ## Setup custom google search tool
    tools = load_tools([
        'google-search'
    ], llm=llm)

    return tools

## An agent has access to an LLM and a suite of tools for example 
## Google Search, Python REPL, math calculator, weather API, etc.
## LLM determines which actions to take and in what order
def setup_agent(tools, llm):
    memory = ConversationBufferMemory(memory_key='chat_history')
    planner = load_chat_planner(llm)
    executor = load_agent_executor(llm, tools, verbose=True)

    ## AgentExecutor chain
    # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # decide a plan of actions to take, and then execute those actions one at a time
    agent = PlanAndExecute(
                planner=planner,
                executor=executor,
                verbose=True,
                reduce_k_below_max_tokens=True
            )
    
    return agent

if __name__ == '__main__':
    query = process_args(sys.argv[1:])

    load_dotenv()

    doc_db = get_db()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

    qa = setup_qa(doc_db, llm)

    tools = setup_tools(qa, llm)

    agent = setup_agent(tools, llm)

    with get_openai_callback() as cb:
        agent.run(query)

    print("\n\n", cb)
