from user_types.base_chat_request import BaseChatRequest


class OpenAIChatRequest(BaseChatRequest):
    model: str = 'gpt-3.5-turbo-16k'
