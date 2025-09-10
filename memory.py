from langchain.memory import ConversationSummaryMemory
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-1.5-flash", temperature=0.2)

summary_memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True,
    memory_key="messages",
    input_key="input"
)
