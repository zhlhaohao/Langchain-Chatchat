from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from server.chat.utils import History
from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import (
    ConversationCallbackHandler,
)


# PPP# 与在线llm模型对话(通过LLMChain) /chat/chat
async def chat(
    query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
    conversation_id: str = Body("", description="对话框ID"),
    history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
    history: Union[int, List[History]] = Body(
        [],
        description="历史对话，设为一个整数可以从数据库中读取历史消息",
        examples=[
            [
                {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                {"role": "assistant", "content": "虎头虎脑"},
            ]
        ],
    ),
    stream: bool = Body(False, description="流式输出"),
    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
    temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
    max_tokens: Optional[int] = Body(
        None, description="限制LLM生成Token数量，默认None代表模型最大值"
    ),
    # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
    prompt_name: str = Body(
        "default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
    ),
):
    """
    这段代码是一个异步函数，用于与OpenAI的聊天模型进行交互。它使用了异步迭代器来返回与模型的对话。函数首先添加一些回调函数到模型中，然后根据提供的历史消息和查询语句构建提示。接着创建了一个OpenAI模型实例，并根据提示和回调函数创建了一个对话链。然后根据是否使用流模式，异步地迭代对话链的结果并返回。最后等待任务完成。

    """

    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        # 负责保存llm response到message db
        message_id = add_message_to_db(  # query:用户输入的问题（字符串）
            chat_type="llm_chat", query=query, conversation_id=conversation_id
        )
        conversation_callback = ConversationCallbackHandler(
            conversation_id=conversation_id,
            message_id=message_id,
            chat_type="llm_chat",
            query=query,
        )
        callbacks.append(conversation_callback)

        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,  #  模型名称: qwen-api
            temperature=temperature,  # 0.7 用户在页面上调整选择
            max_tokens=max_tokens,  # None 代表模型最大值
            callbacks=callbacks,
        )

        if history:  # 优先使用前端传入的历史消息
            # 历史消息数组包括问与答，例如：[{"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"}, {"role": "assistant", "content": "虎头虎脑"}]
            history = [History.from_data(h) for h in history]
            # 获取prompt模板
            prompt_template = get_prompt_template("llm_chat", prompt_name)

            input_msg = History(role="user", content=prompt_template).to_msg_template(
                False
            )
            # 对问题进行模板封装
            input_msg = History(role="user", content=query).to_msg_template()
            # 加上对历史对象消息的封装，形成最终此次对话的输入
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg]
            )
        elif conversation_id and history_len > 0:  # 前端要求从数据库取历史消息
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt)
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBufferDBMemory(
                conversation_id=conversation_id, llm=model, message_limit=history_len
            )
        else:
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(
                False
            )
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        # 生成LLMChain对象
        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # Begin a task that runs in the background.
        task = asyncio.create_task(
            wrap_done(chain.acall({"input": query}), callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "message_id": message_id}, ensure_ascii=False
                )
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id}, ensure_ascii=False
            )

        await task

    """ 
    sse_starlette是一个基于Starlette框架的Python库，用于提供服务器推功能。EventSourceResponse是其中的一个函数，用于创建一个EventSourceResponse对象，该对象可以用于向客户端推送数据。这个函数通常用于实现服务器推功能，其中EventSourceResponse对象可以返回给客户端以实现长轮询（long polling）的效果。
    """
    return EventSourceResponse(chat_iterator())
