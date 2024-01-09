from server.db.session import with_session
import uuid
from server.db.models.conversation_model import ConversationModel

""" 
conversation_id的数据库记录新增
 """


@with_session
def add_conversation_to_db(session, chat_type, name="", conversation_id=None):
    """
    新增聊天记录
    """
    if not conversation_id:
        conversation_id = uuid.uuid4().hex
    c = ConversationModel(id=conversation_id, chat_type=chat_type, name=name)

    session.add(c)
    return c.id
