import logging
import os
import tempfile

from langchain.chains import (
    ConversationalRetrievalChain,
    OpenAIModerationChain,
    SequentialChain,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from utils import MEMORY, load_document
from config_key import set_environment

logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER = logging.getLogger()
set_environment()

def config_retriever(docs: list[Document], use_compression=False, chunk_size=1500):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = 200)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorDB = DocArrayInMemorySearch.from_documents(splits, embeddings)
    retriever = vectorDB.as_retriever(
        search_type='mmr',
        search_kwargs={
            "k": 5,
            "fetch_k": 7,
            "include_metadata": True
        }
    )
    if not use_compression:
        return retriever
    else:
        embeddings_filter = EmbeddingsFilter(
            embeddings=embeddings, similarity_threshold=0.2
        )
        return ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=retriever
        )

def config_chain(retriever: BaseRetriever, temperature=0.1):
    LLM = ChatOpenAI(
        model_name="gpt-4o", temperature=temperature, streaming=True
    )
    MEMORY.output_key = 'answer'
    params = dict(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True
        #max_token_limit=2000
    )
    return ConversationalRetrievalChain.from_llm(**params)
    
def ddg_search_agent(temperature=0.1):
    LLM = ChatOpenAI(
        model_name="gpt-4o", temperature=temperature, streaming=True
    )
    
    tools = load_tools(
        tool_names=['ddg-search'],
        llm=LLM,
        model="gpt-4o-mini"
    )
    return initialize_agent(
        tools=tools, llm=LLM, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
    )
    
def config_retrieval_chain(
        upload_files,
        use_compression=False,
        use_moderation=False,
        use_chunksize=1500,
        use_temperature=0.1,
        use_zeroshoot=False
):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in upload_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        docs.extend(load_document(temp_filepath))
    
    retriever = config_retriever(docs=docs, use_compression=use_compression, chunk_size=use_chunksize)
    chain = config_chain(retriever=retriever, temperature=use_temperature)
    if use_zeroshoot:
        return ddg_search_agent(temperature=use_temperature)
    elif not use_moderation:
        return chain
    else:
        input_variables = ["chat_history", "question"]
        moderation_input = "answer"
        moderation_chain = OpenAIModerationChain(input_key=moderation_input)
        return SequentialChain(
            chains=[chain, moderation_chain],
            input_variables=input_variables
        )

    
