from langchain_community.chat_message_histories import ChatMessageHistory

from user_types.base_chat_request import BaseChatRequest


class RunnableMixin:

    def chat_llm(self, driver, req: BaseChatRequest):
        if req.chat_history:
            llm_chat_history = ChatMessageHistory()
            for event in req.chat_history[-req.conversation_window_size:]:
                if event.event == "user":
                    llm_chat_history.add_user_message(event.text)
                elif event.event == "bot":
                    llm_chat_history.add_ai_message(event.text)
            result = driver.chat_with_history(
                system_prompt=req.system_message_prompt,
                user_message=req.user_message,
                message_history=llm_chat_history,
                rag_content=req.rag_context,
            )
            return result
        else:
            system_skill_prompt = req.system_message_prompt.replace("{", "[").replace("}", "]")
            result = driver.chat(
                system_prompt=system_skill_prompt,
                user_message=req.user_message,
                rag_context=req.rag_context,
            )
            return result
