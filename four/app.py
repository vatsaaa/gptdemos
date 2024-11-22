import ast, json, os, random, requests, sys, time

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.request import urlopen

from newsapi import NewsApiClient

from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

## Project imports
from top_news import NEWSLETTER
from utils import abstract_template, combine_template, news, newsletter_html
from utils import process_args, title_template, top_news_template, urlify

load_dotenv()

def get_url_text(news):
    page = urlopen(news['url'])
    html = page.read().decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    news['text'] = soup.get_text()

'''
To summarize text with a LLM, there are a few strategies. If the whole text fits in the 
context window, then simply feed the raw data and get the result. LangChain calls this
strategy as the “stuff“ chain type.

Often, the number of tokens contained in the text is larger than the LLM's maximum capacity.
A typical strategy is to break down the data into multiple chunks, summarize each chunk, and
summarize the concatenated summaries in a final "combine" step. LangChain refers to this strategy
as “map-reduce“.

Another strategy is to begin the summary with the first chunk and refine it little by little with
each of the following chunks. LangChain refers to this as “refine“ strategy e.g. here is the prompt
template used by LangChain for the refine step.
Your job is to produce a final summary
We have provided an existing summary up to a certain point: {existing_answer}
We have the opportunity to refine the existing summary
(only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary
If the context isn't useful, return the original summary.
'''
def get_text_summary(news, llm):
    text = news['text']

    combine_prompt = PromptTemplate(
        template=combine_template, 
        input_variables=['text']
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=10
    ) 

    # Cast the data as a Document
    # doc = [Document(page_content=text)]

    # we split the data into multiple chunks
    try:
        # docs = char_text_splitter.split_documents(doc)
        docs = text_splitter.create_documents([text])
        combine_chain = load_summarize_chain(
            llm=llm, 
            chain_type='map_reduce',
            combine_prompt=combine_prompt
        )
        news['summary'] = combine_chain.run(docs)
    except Exception as e:
        print("Exception Occurred: ", e)

def generate_image_for_prompt(news, width: int = 1024, height: int = 512):
    prompt = news['summary']
    image_title = news['title']

    data = {
        'key': os.getenv('STABLEDIFFUSION_API_KEY'),
        'width': width,
        'height': height,
        'prompt': prompt,
        "enhance_prompt": "no",
        "safety_checker": "no",
        "samples": "1"
    }

    headers = {
        'Content-type': 'application/json', 
        'Accept': 'text/plain'
    }

    try:
        response = requests.post(
            url=os.getenv('TEXT2IMAGE_URL'), 
            data=json.dumps(data), 
            headers=headers
        )

        if response.json()['status'] != 'error':
            image_url = response.json()['output'][0] 
            img_data = requests.get(image_url).content if image_url else None
        else:
            raise Exception(response.json()['message'])
    except Exception as e:
        print("Exception: ", e, " | Type: ", type(e))

    output_path = None
    if img_data:
        output_path = '/tmp/' + urlify(image_title) + '.png'
        with open(output_path, 'wb') as handler:
            handler.write(img_data)

    news['image']['local'] = output_path
    news['image']['remote'] = image_url

    print("Image: ", news['image']['local'], news['image']['remote'])

def get_all_news(sdate: str, edate: str, topic: str):
    newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
    all_news = newsapi.get_everything(
        q=topic,
        from_param=sdate,
        to=edate,
        sort_by='relevancy',
        language='en'
    )

    return all_news

def generate_newsletter_content(sdate: str, edate: str, topic: str, news_count: int, mock: bool):
    newsletter = {}

    # higher tempratures lead to more varied and surprising output
    # for a much predictable output use temperature value of zero
    llm = ChatOpenAI(model='gpt-3.5-turbo-0613', temperature=0.5)

    # Get all-news about 'artificial intelligence'
    all_news = get_all_news(sdate=sdate, edate=edate, topic=topic)

    # Ask ChatGPT to select the top 10 news from all_news
    TOP_NEWS_PROMPT = PromptTemplate(
        input_variables=['news_count', 'list'],
        template=top_news_template
    )

    title_list = '\n'.join([
        article['title'] 
        for article in all_news['articles']
    ])

    # print(TOP_NEWS_PROMPT.format(news_count=news_count, list=title_list))
    prompt_chain = LLMChain(
        llm=llm, 
        prompt=TOP_NEWS_PROMPT
    )

    top_str = prompt_chain.run({'list': title_list, 'news_count': news_count})

    ## Use abstract syntax grammar to create a python list of top news_count titles
    top_list = ast.literal_eval(top_str)

    # ## Now filter the titles and urls for top 10
    newsletter['articles'] = [
        {
            'title': a['title'],
            'url': a['url']
        }
        for a in all_news['articles']
        if a['title'] in top_list
    ]

    ## For each news in top_news get the text, summary and a suitable image
    for news in newsletter['articles']:
        # get_url_text()
        page = urlopen(news['url'])
        html = page.read().decode('utf-8')
        soup = BeautifulSoup(html, 'html.parser')
        news['text'] = soup.get_text()

        # get_text_summary(news=news, llm=llm)
        combine_prompt = PromptTemplate(
            template=combine_template, 
            input_variables=['text']
        )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=10
        ) 

        # Cast the data as a Document
        # doc = [Document(page_content=text)]

        # we split the data into multiple chunks
        try:
            # docs = char_text_splitter.split_documents(doc)
            docs = text_splitter.create_documents([news['text']])
            combine_chain = load_summarize_chain(
                llm=llm, 
                chain_type='map_reduce',
                combine_prompt=combine_prompt
            )
            news['summary'] = combine_chain.run(docs)
        except Exception as e:
            print("Exception Occurred: ", e)

        # generate_image_for_prompt(news=news)
        width = 1024
        height = 512

        data = {
            'key': os.getenv('STABLEDIFFUSION_API_KEY'),
            'width': width,
            'height': height,
            'prompt': news['summary'],
            "enhance_prompt": "no",
            "safety_checker": "no",
            "samples": "1"
        }

        headers = {
            'Content-type': 'application/json', 
            'Accept': 'text/plain'
        }

        try:
            response = requests.post(
                url=os.getenv('TEXT2IMAGE_URL'), 
                data=json.dumps(data), 
                headers=headers
            )

            if response.json()['status'] != 'error':
                image_url = response.json()['output'][0] 
                img_data = requests.get(image_url).content if image_url else None
            else:
                raise Exception(response.json()['message'])
        except Exception as e:
            print("Exception: ", e, " | Type: ", type(e))

        output_path = None
        if img_data:
            output_path = '/tmp/' + urlify(news['title']) + '.png'
            with open(output_path, 'wb') as handler:
                handler.write(img_data)

        news['image'] = image_url

        print("Image: ", news['image'])
    
    ## Create an abstract for this weekly blog
    ABSTRACT_PROMPT = PromptTemplate(
        input_variables=['sdate', 'edate', 'summaries'],
        template=abstract_template
    )

    summaries = '\n\n\n'.join([
        news['title'] if news['title'] else "" + '\n' + news['summary'] if news['summary'] else ""
        for news in newsletter['articles']
    ])

    # print(ABSTRACT_PROMPT.format(sdate=sdate, edate=edate))
    abstract_chain = LLMChain(
        llm=llm,
        prompt=ABSTRACT_PROMPT
    )

    newsletter['abstract'] = abstract_chain.run({'sdate': sdate, 'edate': edate, 'summaries': summaries})

    print("Abstract: ", newsletter['abstract'])

    ## Create a title for this weekly blog
    TITLE_PROMPT = PromptTemplate(
        input_variables=['sdate', 'edate', 'summaries'], 
        template=title_template
    )

    # print(TITLE_PROMPT.format(sdate=sdate, edate=edate))
    title_chain = LLMChain(
        llm=llm,
        prompt=TITLE_PROMPT
    )

    newsletter['title'] = title_chain.run({'summaries': summaries, 'sdate': sdate, 'edate': edate})

    print(newsletter)

    return newsletter

def get_newslwtter_html(mock, sdate, edate, topic, newsletter):
    count = 0
    newslist_html = ""
    newslist = newsletter['articles']

    for n in newslist:
        if mock:
            time.sleep(random.randint(1, 3))

        count = count + 1
        newslist_html = newslist_html + news.format(
            newsid=count,
            natitle=n['title'],
            naimage=n['image'], naimage_alt="Image Alt",
            article=n['text']
        )

    print(newsletter_html.format(
        title=topic, 
        sdate=sdate, edate=edate,
        nltitle=newsletter['title'],
        nlabstract=newsletter['abstract'],
        news=newslist_html
    ))

def mock_newsletter_content():
    time.sleep(random.randint(1, 3))
    newsletter = NEWSLETTER
    return newsletter

def main(mock: bool, sdate: str, edate: str, topic: str, news_count: int):
    if mock:
        nl = mock_newsletter_content()
    else:
        nl = generate_newsletter_content(sdate=sdate, edate=edate, topic=topic, news_count=news_count, mock=mock)

    get_newslwtter_html(mock=mock, sdate=sdate, edate=edate, topic=topic, newsletter=nl)

if __name__ == '__main__':
    mock, sdate, edate, topic, news_count = process_args(args=sys.argv[1:])

    main(mock=mock, sdate=sdate, edate=edate, topic=topic, news_count=news_count)
