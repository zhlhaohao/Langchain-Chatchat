from langchain.docstore.document import Document
from configs import EMBEDDING_MODEL, logger
from server.model_workers.base import ApiEmbeddingsParams
from server.utils import (
    BaseResponse,
    get_model_worker_config,
    list_embed_models,
    list_online_embed_models,
)
from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from typing import Dict, List

online_embed_models = list_online_embed_models()


def embed_texts(
    texts: List[str],
    embed_model: str = EMBEDDING_MODEL,
    to_query: bool = False,
) -> BaseResponse:
    """
    # PPP# 对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    TODO: 也许需要加入缓存机制，减少 token 消耗
    """
    try:
        if embed_model in list_embed_models():  # 使用本地Embeddings模型
            from server.utils import load_local_embeddings

            # 加载本地嵌入模型
            embeddings = load_local_embeddings(model=embed_model)
            # 对文本进行嵌入化，返回二维的数组[一个文件切分后的段落数，嵌入向量维度] ,调用HuggingFaceBgeEmbeddings.embed_documents(),最后再调用SentenceTransformer.enocde()进行向量化编码，SentenceTransformer使用了Bert等预训练模型，它是bge-reranker-large所使用的模型
            return BaseResponse(data=embeddings.embed_documents(texts))

        if embed_model in list_online_embed_models():  # 使用在线API进行向量化
            config = get_model_worker_config(embed_model)
            worker_class = config.get("worker_class")
            embed_model = config.get("embed_model")
            worker = worker_class()
            if worker_class.can_embedding():
                params = ApiEmbeddingsParams(
                    texts=texts, to_query=to_query, embed_model=embed_model
                )
                resp = worker.do_embeddings(params)
                return BaseResponse(**resp)

        return BaseResponse(
            code=500, msg=f"指定的模型 {embed_model} 不支持 Embeddings 功能。"
        )
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


async def aembed_texts(
    texts: List[str],
    embed_model: str = EMBEDDING_MODEL,
    to_query: bool = False,
) -> BaseResponse:
    """
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    """
    try:
        if embed_model in list_embed_models():  # 使用本地Embeddings模型
            from server.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=await embeddings.aembed_documents(texts))

        if embed_model in list_online_embed_models():  # 使用在线API
            return await run_in_threadpool(
                embed_texts, texts=texts, embed_model=embed_model, to_query=to_query
            )
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


def embed_texts_endpoint(
    texts: List[str] = Body(
        ..., description="要嵌入的文本列表", examples=[["hello", "world"]]
    ),
    embed_model: str = Body(
        EMBEDDING_MODEL,
        description=f"使用的嵌入模型，除了本地部署的Embedding模型，也支持在线API({online_embed_models})提供的嵌入服务。",
    ),
    to_query: bool = Body(
        False,
        description="向量是否用于查询。有些模型如Minimax对存储/查询的向量进行了区分优化。",
    ),
) -> BaseResponse:
    """
    对文本进行向量化，返回 BaseResponse(data=List[List[float]])
    """
    return embed_texts(texts=texts, embed_model=embed_model, to_query=to_query)


def embed_documents(
    docs: List[Document],
    embed_model: str = EMBEDDING_MODEL,
    to_query: bool = False,
) -> Dict:
    """
    # PPP# 将 List[Document] 向量化，转化为 VectorStore.add_embeddings 可以接受的参数
    """
    texts = [x.page_content for x in docs]  # 文本内容
    metadatas = [x.metadata for x in docs]  # 文件名
    embeddings = embed_texts(
        texts=texts, embed_model=embed_model, to_query=to_query
    ).data
    if (
        embeddings is not None
    ):  # emdeddings是二维数组 - [段落数,向量维度],最后一维是将该段落embedding后的向量值
        return {
            "texts": texts,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
