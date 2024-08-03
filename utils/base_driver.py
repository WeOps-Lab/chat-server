from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from loguru import logger


class BaseDriver:
    def _log_result(self, user_message, system_prompt, result):
        logger.info(f"""
            请求消息: {user_message}
            系统消息: {system_prompt}
            响应消息: {result.content}
            输入令牌: {result.usage_metadata['input_tokens']}
            输出令牌: {result.usage_metadata['output_tokens']}
            总令牌: {result.usage_metadata['total_tokens']}
        """)

    def _handle_exception(self, e):
        logger.error(f"聊天出错: {e}")
        return "服务端异常"

    def chat(self, system_prompt, user_message, rag_context=""):
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{system_prompt}, Here is some context: {rag_context}"),
                ("human", "{text}"),
            ])

            chain = prompt | self.client
            result = chain.invoke({"text": user_message})

            self._log_result(user_message, system_prompt, result)
            return result.content
        except Exception as e:
            return self._handle_exception(e)

    def chat_with_history(self, system_prompt, user_message, message_history, rag_content=""):
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"{system_prompt}, Here is some context: {rag_content}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            chain = prompt | self.client
            chain_with_history = RunnableWithMessageHistory(
                chain,
                get_session_history=lambda: message_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            result = chain_with_history.invoke({"input": user_message})

            self._log_result(user_message, system_prompt, result)
            return result.content
        except Exception as e:
            return self._handle_exception(e)
