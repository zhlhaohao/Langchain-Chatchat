from server.db.session import with_session
from typing import Dict, List
import uuid
from server.db.models.message_model import MessageModel

""" 
# PPP# 聊天记录的数据库操作（新增、更新、根据conversation_id查询、根据message_id查询）
 """


@with_session
def add_message_to_db(
    session,
    conversation_id: str,
    chat_type,
    query,
    response="",
    message_id=None,
    metadata: Dict = {},
):
    """
    新增聊天记录
    """
    if not message_id:
        message_id = uuid.uuid4().hex
    m = MessageModel(
        id=message_id,
        chat_type=chat_type,
        query=query,
        response=response,
        conversation_id=conversation_id,
        meta_data=metadata,
    )
    session.add(m)
    session.commit()
    return m.id


@with_session
def update_message(session, message_id, response: str = None, metadata: Dict = None):
    """
    更新已有的聊天记录
    """
    m = get_message_by_id(message_id)
    if m is not None:
        if response is not None:
            m.response = response
        if isinstance(metadata, dict):
            m.meta_data = metadata
        session.add(m)
        session.commit()
        return m.id


@with_session
def get_message_by_id(session, message_id) -> MessageModel:
    """
    查询聊天记录
    """
    m = session.query(MessageModel).filter_by(id=message_id).first()
    return m


@with_session
def feedback_message_to_db(session, message_id, feedback_score, feedback_reason):
    """
    反馈聊天记录
    """
    m = session.query(MessageModel).filter_by(id=message_id).first()
    if m:
        m.feedback_score = feedback_score
        m.feedback_reason = feedback_reason
    session.commit()
    return m.id


# PPP## 根据conversation_id查询聊天消息记录
@with_session
def filter_message(session, conversation_id: str, limit: int = 10):
    """
    使用数据库会话从指定对话ID中获取最新的、有回复的消息记录，并将其转换为字典列表。

    参数：
    session (Session): SQLAlchemy的数据库会话对象，用于执行SQL查询。
    conversation_id (str): 要过滤消息的对话ID。
    limit (int): 可选参数，默认值为10，表示要返回的最新消息条数。

    过程：
    1. 执行SQL查询以获取与指定对话ID关联且响应（response）不为空的所有`MessageModel`记录，
       并按创建时间（create_time）降序排列。
    2. 查询结果限制在最新的limit条记录以内。
    3. 将查询结果中的每一项转化为一个包含查询内容（query）和回复内容（response）的字典，并将这些字典添加到一个新的列表中。

    返回值：
    data (List[Dict[str, str]]): 包含了最近limit条对话记录的字典列表，每个字典有两个键："query"和"response"。
    """

    # 构建SQL查询语句，筛选出指定对话ID下非空回复的消息记录，并按创建时间降序排序
    messages = (
        session.query(MessageModel)
        .filter_by(conversation_id=conversation_id)
        .filter(
            MessageModel.response != ""
        )  # 用户最新的query 也会插入到db，忽略这个message record
        .order_by(MessageModel.create_time.desc())  # 按照创建时间降序排列
        .limit(limit)  # 取最新limit条记录
        .all()  # 执行查询并获取所有匹配的结果
    )
    data = []

    # 遍历查询结果，将每一条MessageModel对象的信息转换为字典，并添加到data列表中
    for m in messages:
        data.append({"query": m.query, "response": m.response})

    # 返回包含最新limit条消息记录的字典列表
    return data
