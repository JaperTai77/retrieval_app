# Chatbot Using RAG and Web Search

A user interface that can chat with bot using self-uploaded documents or search for information online using DuckDuckGo. Created using langchain and OpenAI API.

## Environment
1. Install Python
  - Visit [Python](https://www.python.org/) to install Python
2. Create Virtual Environment
  - Install virtual environment package
```
pip install virtualenv
```
  - Create virtual environment
```
virtualenv [your_env_name]
```
  - Activate environment
```
Script\activate
```
  
3. Install Package
```
pip install -r requirements.txt
```

4. Set up config_key.py to store openai api key
     - create a config_key.py file in directory.
     - create a openai account and set up a key, the key can be create and found [here](https://platform.openai.com/api-keys)
     - copy the following code with your own key and paste it in config_key.py
```
import os

def set_environment():
     os.environ['OPENAI_API_KEY']='sk**'
```

## Run the App
Now run the streamlit command to start the service
```
streamlit run app.py 
```
The web will be on http://localhost:8501/ if run on own computer or server. Now you can chat with your own text and pdf document or search question on web.

Upload a document.
![chatbot_demo_1](https://github.com/user-attachments/assets/417263db-cb19-4706-8fc7-3244328f9a0d)

Chat with document or search on web.
![chatbot_demo_2](https://github.com/user-attachments/assets/7bdef658-8174-4c70-9bdc-097c688f2879)

When done deactivate environment.
```
deactivate
```

## Credit
Repo inspired by [benman1/generative_ai_with_langchain](https://github.com/benman1/generative_ai_with_langchain?tab=readme-ov-file)
