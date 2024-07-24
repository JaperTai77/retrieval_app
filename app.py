import streamlit as st
import logging
from utils import MEMORY, DocumentLoader
from chat import config_retrieval_chain
from streamlit.external.langchain import StreamlitCallbackHandler


logging.basicConfig(encoding="utf-8", level=logging.INFO)
LOGGER = logging.getLogger()

st.set_page_config(page_title="文件", page_icon="👻")
st.title("👻文件問答👻")

uploaded_files = st.sidebar.file_uploader(
    label="上傳文件🐣",
    type=list(DocumentLoader.supported_extensions.keys()),
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("請上傳附件🐣")
    st.stop()

use_chunk = st.sidebar.slider(
    '文章切割',
    500, 2000, (1000)
)
use_temperature = st.sidebar.slider(
     '溫度 (越高越有創意🦄)',
     0.0, 1.0, (0.1))

use_compression = st.checkbox("Compression🛠️(上傳文件)", value=False)
use_moderation = st.checkbox("Moderation🕸️(過濾不合宜回答)", value=False)
use_ddg_search = st.checkbox("使用DuckDuckGO搜尋🦆(不會使用上傳文件)", value=False)

CONV_CHAIN = config_retrieval_chain(
    uploaded_files,
    use_compression=use_compression,
    use_moderation=use_moderation,
    use_chunksize=use_chunk,
    use_temperature=use_temperature,
    use_zeroshoot=use_ddg_search
)

if st.sidebar.button("清除對話和占存對話🦭"):
    MEMORY.chat_memory.clear()

avatars = {"human": "user", "ai": "assistant"}

if len(MEMORY.chat_memory.messages) == 0:
    st.chat_message("assistant").markdown("請提問🤖")

assistant = st.chat_message("assistant")
if user_query := st.chat_input(placeholder="說點什麼"):
    st.chat_message("user").write(user_query)
    container = st.empty()
    stream_handler = StreamlitCallbackHandler(container)
    with st.chat_message("assistant"):
        if use_ddg_search:
            response = CONV_CHAIN.invoke(
                {"input": user_query}, {"callbacks": [stream_handler]}
            )
            st.write(response["output"])
        else:
            params = {
                "question": user_query,
                "chat_history": MEMORY.chat_memory.messages,
            }
            response = CONV_CHAIN.run(params, callbacks=[stream_handler])
            if response:
                container.markdown(response)