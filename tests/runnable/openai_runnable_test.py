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
    rag_context = """凯瑟喵，Kaisersama，95后唱作人，海外音乐系学生。代表作品有《撒野》《谎》《一分之二》《Midnight》等。 “将幻想塞给他们，将他们带入新的森林。”"""
    req = OpenAIChatRequest(
        system_message_prompt="你是一个严谨的问答助手，会根据我提供的背景信息进行简洁的问答，不能捏造任何信息",
        openai_api_base=openai_base_url,
        model="gpt-4o-mini",
        openai_api_key=openai_key,
        rag_context=rag_context,
        user_message="介绍一下凯瑟喵",
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
