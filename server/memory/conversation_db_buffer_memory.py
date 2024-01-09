import logging
from typing import Any, List, Dict

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string, BaseMessage, HumanMessage, AIMessage
from langchain.schema.language_model import BaseLanguageModel
from server.db.repository.message_repository import filter_message
from server.db.models.message_model import MessageModel


# PPP#历史聊天消息数据库管理的封装
class ConversationBufferDBMemory(BaseChatMemory):
    conversation_id: str
    human_prefix: str = "Human"  # 默认的人类消息前缀
    ai_prefix: str = "Assistant"  # 默认的AI消息前缀
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000
    message_limit: int = 10

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        # fetch limited messages desc, and return reversed
        """从数据库中获取属于conversation_id的消息并转换为消息列表"""
        messages = filter_message(
            conversation_id=self.conversation_id, limit=self.message_limit
        )
        # 返回的记录按时间倒序，转为正序
        messages = list(reversed(messages))
        chat_messages: List[BaseMessage] = []
        for message in messages:
            # 将每条人类消息添加到缓冲区
            chat_messages.append(HumanMessage(content=message["query"]))
            # 将每条AI回复消息添加到缓冲区
            chat_messages.append(AIMessage(content=message["response"]))

        if not chat_messages:
            return []

        # 如果缓冲区总字符数超过最大限制，则进行裁剪
        curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit and chat_messages:
                pruned_memory.append(chat_messages.pop(0))
                curr_buffer_length = self.llm.get_num_tokens(
                    get_buffer_string(chat_messages)
                )

        return chat_messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        # PPP## 从数据库中获取指定conversation_id的聊天的历史消息，并将其格式化后存储字符串中返回。

        参数：
        inputs (Dict[str, Any]): 输入参数字典，此处未使用

        返回值：
        Dict[str, Any]: 一个字典，其中键为`memory_key`（默认为"history"），值为格式化后的历史消息缓冲区。

        过程：
        1. 调用`self.buffer`属性方法获取历史消息列表。
        2. 检查类实例的`return_messages`属性。如果为True，则直接将原始消息列表作为缓冲区内容返回。
        3. 如果`return_messages`为False，则使用`get_buffer_string`函数将消息列表转换为字符串格式，
           并添加人类和AI消息前缀。这将生成一个格式化的、可由语言模型理解的文本缓冲区。
        4. 将格式化后的缓冲区内容放入一个字典中，键为`history`，并返回该字典。
        """
        buffer: Any = self.buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            """ 
            convert sequence of Messages to strings and concatenate them into one string.
            """
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,  # 定制人类消息前缀
                ai_prefix=self.ai_prefix,  # 定制AI消息前缀
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed"""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass
