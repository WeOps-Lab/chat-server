import os

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from utils.openai_driver import OpenAIDriver
from loguru import logger

load_dotenv()
openai_base_url = os.getenv('OPENAI_BASE_URL')
openai_key = os.getenv('OPENAI_API_KEY')


def test_openai_chat():
    driver = OpenAIDriver(openai_key, openai_base_url, 0.7, 'gpt-4o-mini')
    result = driver.chat("", "你好呀")
    logger.info(result)


def test_openai_chat_with_history():
    driver = OpenAIDriver(openai_key, openai_base_url, 0.7, 'gpt-4o-mini')
    llm_chat_history = ChatMessageHistory()
    llm_chat_history.add_user_message("凯瑟喵有什么好听的歌")
    llm_chat_history.add_ai_message("小雏菊、安慰剂都是不错的哦")
    llm_chat_history.add_user_message("How Sweet是她的歌吗")
    llm_chat_history.add_ai_message("不是的，是NewJeans的歌哟")
    result = driver.chat_with_history(
        "扮演专业的小助手",
        "我们聊了什么",
        llm_chat_history,
        ""
    )
    logger.info(result)
