import re
import getopt

def urlify(s: str):

    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)

    # Replace all runs of whitespace with a single dash
    s = re.sub(r"\s+", '-', s)

    return s

def usage():
    print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")
    print('Usage: python3 app.py -h --help: Prints this help message')
    print('Usage: python3 app.py -a <blog topic> -f <from date> -t<to date>')
    print('Usage: python3 app.py -a <blog topic> -f <from date> -t<to date> -m: Use mock news')

    '''
    Add help messages for more options here
    '''

    return

def process_args(args):
    mock = False
    edate = None
    sdate = None
    topic = None
    count = None

    try:
        opts, args = getopt.getopt(args, "a:c:f:hmt:",
                                   [
                                       "about",
                                       "count",
                                       "from",
                                       "help",
                                       "mock",
                                       "to"
                                   ])
    except getopt.GetoptError as err:
        print("Error: " + str(err))
        usage()
        exit(-1)

    ## All options to process must be added here
    ## only in alphabetical order, refer to elif
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            exit(-2)
        elif o in ("-m", "--mock"):
            mock = True
        elif o in ("-a", "--about"):
            topic = a
        elif o in ("-c", "--count"):
            count = a if a else 3
        elif o in ("-f", "--from"):
            sdate = a
        elif o in ("-t", "--to"):
            edate = a
        else:
            assert False, "Invalid option: " + o + "provided!"

    if not topic:
        print("Usage: python3 app -a option is mandatory to specify")
        exit(-3)

    if not mock:
        if (sdate and edate == None) or (edate and sdate == None):
            print('Usage: python3 app.py -f and -t options are required together!')
            exit(-4)
    
    ## Return whatever is used by functions
    ## in main() Else just return empty
    return mock, sdate, edate, topic, count

newsletter_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
</head>
<body>
    <h1>{sdate} to {edate}: {nltitle}</h1>
    <h3>{nlabstract}</h3>
    {news}
</body>
</html>
'''

news = '''
    <div class="news_article" id="{newsid}">
        </br>
        <h4>{natitle}</h4>
        <img src={naimage} alt={naimage_alt}/>
        <br/> <div align="justify"> {article}</div>
    </div>
'''

abstract_template = """
Use the following article summaries to generate a 2 or 3 sentences abstract for a blog about Machine Learning. 
Those articles are the most impactful news of the past week from {sdate} to {edate}  

{summaries}
"""

top_news_template = """
Extract from the following list the {news_count} most important news about machine learning. Return the answer as a python list. 

For example: ['text 1', 'text 2', 'text 3']

{list}
"""

title_template = """
Use the following article summaries to generate a title for a blog about Machine Learning.
These articles are the most impactful news of the past week from {sdate} to {edate}.
The title needs to be appealing such that people are excited to read the blog.

{summaries}
"""

combine_template = """
Your job is to produce a concise summary of a news article about Machine Learning
Provide the important points and explain why they matter
The summary is for a blog so it has to be exciting to read

Write a concise summary of the following:

{text}

CONCISE SUMMARY:
"""
