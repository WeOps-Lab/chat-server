import os

from dotenv import load_dotenv

from runnable.openai_runnable import OpenAIRunnable
from user_types.chat_history import ChatHistory
from user_types.openai_chat_request import OpenAIChatRequest
from utils.openai_driver import OpenAIDriver
from loguru import logger

load_dotenv()
openai_base_url = os.getenv('OPENAI_BASE_URL')
openai_key = os.getenv('OPENAI_API_KEY')


def test_openai_chat():
    runnable = OpenAIRunnable()
    req = OpenAIChatRequest(
        system_message_prompt="",
        openai_api_base=openai_base_url,
        openai_api_key=openai_key,
        user_message="你好呀",
    )
    result = runnable.openai_chat(req)
    logger.info(result)


def test_openai_chat_with_history():
    runnable = OpenAIRunnable()
    req = OpenAIChatRequest(
        system_message_prompt="扮演小助手",
        openai_api_base=openai_base_url,
        openai_api_key=openai_key,
        user_message="我们聊了什么",
        conversation_window_size=2,
        model="gpt-4o-mini",
        chat_history=[
            ChatHistory(event="user", text="凯瑟喵有什么好听的歌"),
            ChatHistory(event="bot", text="小雏菊、安慰剂都是不错的哦，蓝鲸也很棒哦"),
            ChatHistory(event="user", text="How Sweet是她的歌吗"),
            ChatHistory(event="bot", text="不是的，是NewJeans的歌哟"),
        ]
    )
    result = runnable.openai_chat(req)
    logger.info(result)
