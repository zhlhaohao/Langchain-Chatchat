from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (
    LLM_MODELS,
    VECTOR_SEARCH_TOP_K,
    SCORE_THRESHOLD,
    TEMPERATURE,
    USE_RERANKER,
    RERANKER_MODEL,
    RERANKER_MAX_LENGTH,
    MODEL_PATH,
)
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
import json
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device


# PPP# 基于知识库的对话
async def knowledge_base_chat(
    query: str = Body(..., description="用户输入", examples=["你好"]),
    knowledge_base_name: str = Body(
        ..., description="知识库名称", examples=["samples"]
    ),
    top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
    score_threshold: float = Body(
        SCORE_THRESHOLD,
        description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
        ge=0,
        le=2,
    ),
    history: List[History] = Body(
        [],
        description="历史对话",
        examples=[
            [
                {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                {"role": "assistant", "content": "虎头虎脑"},
            ]
        ],
    ),
    stream: bool = Body(False, description="流式输出"),
    model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
    temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(
        None, description="限制LLM生成Token数量，默认None代表模型最大值"
    ),
    prompt_name: str = Body(
        "default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
    ),
    request: Request = None,
):
    # 返回KBService知识库对象（根据名称如samples,向量库类型以及嵌入模型）
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    # PPP## 与知识库对话，用枚举的方式实现流式输出
    # 定义一个异步迭代器函数，用于从知识库中获取与查询相关的答案并生成聊天回复。
    async def knowledge_base_chat_iterator(
        query: str,
        top_k: int,  # 指定从知识库中检索相关文档的前top_k个结果
        history: Optional[List[History]],  # 用户对话历史记录
        model_name: str = model_name,  # 指定对话LLM模型名称
        prompt_name: str = prompt_name,  # 指定prompt模板名称
    ) -> AsyncIterable[str]:
        nonlocal max_tokens  # 引用外部作用域中的max_tokens变量

        # 初始化大模型的回答结果的回调处理器
        callback = AsyncIteratorCallbackHandler()

        # 如果max_tokens设置为非正整数，则重置其值为None
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        # 获取指定参数的对话LLM模型实例
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        # 在线程池中异步执行search_docs函数，根据用户输入在知识库中搜索最相关的top_k个段落
        docs = await run_in_threadpool(
            search_docs,
            query=query,
            knowledge_base_name=knowledge_base_name,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # 如果启用了reranker，则使用reranker对检索结果进行重新排序
        if USE_RERANKER:
            reranker_model_path = MODEL_PATH["reranker"].get(
                RERANKER_MODEL, "BAAI/bge-reranker-large"
            )
            reranker_model = LangchainReranker(
                top_n=top_k,
                device=embedding_device(),
                max_length=RERANKER_MAX_LENGTH,
                model_name_or_path=reranker_model_path,
            )
            docs = reranker_model.compress_documents(documents=docs, query=query)

        # 将所有检索出来的相关文档的内容合并成一个字符串上下文（context）
        context = "\n".join([doc.page_content for doc in docs])

        # 如果检索不到，选择empty模板
        if not docs:
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:  # 选择指定的知识库问答模板
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)

        # 创建用户消息的历史记录，并将其格式化为输入消息
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg]
        )

        """ 
        创建LLM链对象以处理上下文和问题
        chat_prompt: 包含N条历史用户提问与AI助手回答，最后1条是提示模板template:
        '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n<已知信息>{{ context }}</已知信息>\n<问题>{{ question }}</问题>\n'
        """
        chain = LLMChain(prompt=chat_prompt, llm=model)

        print(f"问题:{query},已知信息:\n{context}")
        # 创建后台任务来运行LLM模型推理，context填充到模板context，query填充到模板question
        # chain.acall() 调用大模型接口得到回答
        task = asyncio.create_task(
            wrap_done(
                chain.acall({"context": context, "question": query}), callback.done
            )
        )

        # 构建文档来源信息列表，包括文件名、链接和内容 --- 返回到页面
        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode(
                {"knowledge_base_name": knowledge_base_name, "file_name": filename}
            )
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = (
                f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            )
            source_documents.append(text)

        # 若未找到相关文档，添加提示信息
        if not source_documents:
            source_documents.append(
                "<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>"
            )

        # 根据stream参数决定是否采用服务器发送事件方式流式响应
        if stream:
            # 大模型每返回一批token，Langchain会调用callback yield tokens
            async for token in callback.aiter():
                # 使用server-sent-events逐块返回回答给前端页面
                yield json.dumps({"answer": token}, ensure_ascii=False)

            # 最后返回文档来源信息
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            # 一次性返回完整回答和文档来源信息
            yield json.dumps(
                {"answer": answer, "docs": source_documents}, ensure_ascii=False
            )

        # 等待后台任务完成
        await task

    # 最后，返回一个EventSourceResponse对象，使用上述定义的异步迭代器函数
    return EventSourceResponse(
        knowledge_base_chat_iterator(query, top_k, history, model_name, prompt_name)
    )
