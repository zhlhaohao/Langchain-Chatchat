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
    # 定义一个异步生成器函数，用于实现与OpenAI模型的对话交互，并将对话内容yield返回给客户端
    # query:用户输入的问题（字符串）
    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens  # 引用外部作用域中的history和max_tokens变量

        # AsyncIteratorCallbackHandler是LangChain自己的回调处理器，必须作为回调链的第1个
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        # 将用户输入的问题存入数据库
        message_id = add_message_to_db(
            chat_type="llm_chat", query=query, conversation_id=conversation_id
        )

        # 添加对话管理的回调函数到回调链中: 捕获每次模型返回的响应，保存到数据库或某种持久化存储中，以便于后续追踪对话历史。
        conversation_callback = ConversationCallbackHandler(
            conversation_id=conversation_id,
            message_id=message_id,
            chat_type="llm_chat",
            query=query,
        )
        callbacks.append(conversation_callback)

        # 处理max_tokens参数，若小于等于0则设置为None（代表模型默认最大值）
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        # 获取此次对话所使用的ChatOpenAI模型实例
        model = get_ChatOpenAI(
            model_name=model_name,  # 模型名称: 例如qwen-api
            temperature=temperature,  # 用户在页面上可调整的温度参数（控制生成结果随机性）
            max_tokens=max_tokens,  # 最大生成token数量或使用模型默认值
            callbacks=callbacks,  # 传入回调链
        )

        # 根据前端传入的历史消息构建prompt
        if history:  # 如果前端提供了历史消息
            # 历史消息数组包括问与答，例如：[{"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"}, {"role": "assistant", "content": "虎头虎脑"}]
            history = [History.from_data(h) for h in history]  # 转换历史消息数据结构
            prompt_template = get_prompt_template(
                "llm_chat", prompt_name
            )  # 获取prompt模板
            # 对新问题进行模板封装
            input_msg = History(role="user", content=prompt_template).to_msg_template(
                False
            )
            # 结合历史消息和当前问题构建完整的对话输入prompt
            input_msg = History(role="user", content=query).to_msg_template()
            # 加上对历史对象消息的封装，形成最终此次对话的输入
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg]
            )
        elif conversation_id and history_len > 0:  # 若前端要求从数据库取历史消息
            # 使用带有记忆功能的prompt模板
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt)

            # 创建内存对象，用于存储并提供对话历史给模型
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBufferDBMemory(
                conversation_id=conversation_id, llm=model, message_limit=history_len
            )
        else:  # 若没有历史消息，则直接使用基础prompt模板
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(
                False
            )
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        # 创建LLMChain对象，用于执行模型调用及管理对话上下文
        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # 启动一个异步任务来执行模型调用，并在完成后触发LangChain的自己的回调函数
        task = asyncio.create_task(
            wrap_done(chain.acall({"input": query}), callback.done),
        )

        # 实时流式返回模型生成的每个token（服务器推送事件SSE）
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "message_id": message_id}, ensure_ascii=False
                )
        else:  # 非实时模式下，等待所有token生成完毕后合并为完整回答再返回
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id}, ensure_ascii=False
            )
        # 确保异步任务完成
        await task

    """ 
    sse_starlette是一个基于Starlette框架的Python库，用于提供服务器推功能。EventSourceResponse是其中的一个函数，用于创建一个EventSourceResponse对象，该对象可以用于向客户端推送数据。这个函数通常用于实现服务器推功能，其中EventSourceResponse对象可以返回给客户端以实现长轮询（long polling）的效果。
    """
    # 返回包装了chat_iterator的EventSourceResponse对象以支持服务器推送事件（SSE）功能
    return EventSourceResponse(chat_iterator())
