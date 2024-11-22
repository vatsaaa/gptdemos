from bs4 import BeautifulSoup
from dotenv import load_dotenv
from enum import Enum
import feedparser, getopt, instabot, json
from os import getenv, path, rename
import requests
from pprint import pprint
import shutil
from sys import argv, stderr
import tweepy

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI, ChatAnthropic

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from utils import urlify

class InstaPostType(Enum):
    IMAGE = 1
    VIDEO = 2

def post_instagram_pic(pic_path: str, caption: str):
    bot = instabot.Bot()

    bot.login(username=getenv('INSTAGRAM_USER'), password=getenv('INSTAGRAM_USER_PASS'))
    bot.upload_photo(pic_path, caption=caption)

def generate_image_for_prompt(news: dict, width: int = 1024, height: int = 512):
    prompt = news['article']
    image_title = news['title']

    img_data = None
    image_url = None
    output_path = None

    data = {
        'key': getenv('STABLEDIFFUSION_API_KEY'),
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
            url=getenv('TEXT2IMAGE_URL'), 
            data=json.dumps(data), 
            headers=headers
        )

        if response.json()['status'] != 'error':
            image_url = response.json()['output'][0] 
            img_data = requests.get(image_url).content if image_url else None
        else:
            raise Exception(response.json()['message'])
    except Exception as e:
        print("Exception: ", e, " | Type: ", type(e), file=stderr)

    if img_data:
        output_path = './images/' + urlify(image_title) + '.png'
        with open(output_path, 'wb') as handler:
            handler.write(img_data)

    return image_url

def shorten_url(long_url: str, access_token: str) -> str:
    api_url = getenv("SHORTENER_URL_1")
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    data = {
        "long_url": long_url
    }
    
    response = requests.post(api_url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()["link"]

def build_tweet(text: str, link: str) -> str:
    tweet = None
    docs = None
    llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=0.5)

    template_prompt = """
    {text}

    Please suggest in only one line what does above text do. 
    Respond in first person. The response must be it catchy, engaging and suitable for a single tweet.
    Do include link {link} in the response.
    Please sparingly use phrase 'Dive into the', instead use similar catchy and appealing phrases
    """

    prompt_template = PromptTemplate(
        template=template_prompt
        , input_variables=['text', 'link']
    )

    tweet_prompt = prompt_template.format(text=text, link=link)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=10
    )

    # we split the data into multiple chunks
    try:
        docs = text_splitter.create_documents([text, link])
        combine_chain = load_summarize_chain(
            llm=llm, 
            chain_type='stuff'
        )
        tweet = combine_chain.run(docs)
    except Exception as e:
        print("Exception Occurred: ", e)
        exit(2)

    print(tweet.strip())

    return tweet

def main(main_args: dict):
    feed = feedparser.parse(getenv('BLOGSPOT_URL'))

    items = []
    image_url = None

    for entry in feed.entries:
        article_html = entry.summary_detail.value
        link = shorten_url(long_url=entry.link, access_token=getenv("SHORTENER_TOKEN_1"))

        soup = BeautifulSoup(article_html, 'html.parser')
        article_text = soup.get_text().replace('\xa0', '').replace('\n', ' ').strip()

        if main_args['image']:
            image_url = generate_image_for_prompt({'article': article_text, 'title': entry.title})

        item = {
            'id': entry.id,
            'link': link,
            'article': "{dq}{at}{dq}".format(at=article_text, dq='\"'), 
            'image': image_url
        }

        item['tweet'] = build_tweet(text=item.get('article'), link=item.get('link')) if main_args.get('build') else None

        items.append(item)

    return items

def usage(exit_code: int):
    print("Usage: python app.py [OPTIONS]")
    print("Options:")
    print("\t-h, --help\t\tThis usage message...")
    print("\t-i, --image\t\tGenerate image for prompt")
    exit(exit_code)

def process_args(args: list):
    retvals = {
        'build': False,
        'image': False,
        'tweet': False
    }

    try:
        opts, args = getopt.getopt(args, "bg:hit", ["build", "gram=", "help", "image", "tweet"])
    except getopt.GetoptError as err:
        print(err)
        exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage(2)
        elif opt in ("-b", "--build"):
            retvals['build'] = True
        elif opt in ("-g", "--gram"):
            retvals['gram'] = True
            
            file_extension = path.splitext(arg)[1].lower()
            if file_extension in ['.jpg', '.gif']:
                return 'IMAGE'
            elif file_extension == '.mp4':
                return 'VIDEO'
            else:
                print("Error: File type can be IMAGE (.JPG, JPEG, .PNG, .GIF) or VIDEO (.MP4))", file=stderr)
                usage(3)
        elif opt in ("-i", "--image"):
            retvals['image'] = True
        elif opt in ("-t", "--tweet"):
            retvals['tweet'] = True
            retvals['build'] = True # tweet has to be built, to post it
        else:
            usage(4)

    return retvals

def post_tweet(message: str):
    if not message:
        return
    
    consumer_key = getenv('TWITTER_API_KEY')
    consumer_secret = getenv('TWITTER_API_KEY_SECRET')
    access_token = getenv('TWITTER_ACCESS_TOKEN')
    access_token_secret = getenv('TWITTER_ACCESS_TOKEN_SECRET')

    client = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )

    client.create_tweet(text=message)

if __name__ == '__main__':
    load_dotenv()
    margs = process_args(argv[1:])
    records = main(margs)

    try:
        file1 = open("myfile.json", "a")
    except FileExistsError:
        print("Error: File does not exist")
    else:
        for record in records:
            json.dump(record, file1)
            file1.write(",\n\n")

            post_tweet(record.get('tweet')) if margs.get('tweet') else None
    finally:
        file1.close()
