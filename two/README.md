https://discuss.streamlit.io/t/how-to-use-multiple-files-loaded-with-file-uploader/25051

https://blog.streamlit.io/introducing-multipage-apps/

https://clemenssiebler.com/posts/using-langchain-and-chatgpt-turbo-with-azure-openai-service/

Giving access to Google Search

Let’s give ChatGPT access to Google Search as. We need the API key for. Follow those steps to get those:

Go to the Google Cloud Console.
If you don't already have an account, create one and log in
Create a new project by clicking on the Select a Project dropdown at the top of the page and clicking New Project
Give it a name and click Create
Set up a custom search API and add to your .env file:
Go to the APIs & Services Dashboard
Click Enable APIs and Services
Search for Custom Search API and click on it
Click Enable
Go to the Credentials page
Click Create Credentials
Choose API Key
Copy the API key
Enable the Custom Search API on your project. (Might need to wait few minutes to propagate.) Set up a custom search engine and add to your .env file:
Go to the Custom Search Engine page
Click Add
Set up your search engine by following the prompts. You can choose to search the entire web or specific sites
Once you've created your search engine, click on Control Panel
Click Basics
Copy the Search engine ID
Now we just need to set the environment variables and import the tools