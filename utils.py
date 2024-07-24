import logging
import pathlib
from typing import Any
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document


def init_memory():
    """Initialize the memory for contextual conversation.

    We are caching this, so it won't be deleted
     every time, we restart the server.
     """
    return ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
MEMORY = init_memory()

class DocumentLoaderException(Exception):
    pass

class DocumentLoader(object):
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader
    }

def load_document(temp_filepath: str) -> list[Document]:
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extensions.get(ext)
    if not loader:
        raise DocumentLoaderException(
            f"檔案格式不合 <{ext}>"
        )

    loaded = loader(temp_filepath)
    docs = loaded.load()
    logging.info(docs)
    return docs