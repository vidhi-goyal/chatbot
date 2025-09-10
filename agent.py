from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool as LC_Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from google.oauth2 import service_account
import vertexai
from rag_utils import load_vector_store
from langchain.chains import RetrievalQA
from cache_memory import check_cache, update_cache
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from utilis import get_tools_and_model
from cache_memory import embed_query, cosine_similarity
# === Vertex Setup ===
credentials = service_account.Credentials.from_service_account_file(
    r"C:\Users\prasukh.jain\Desktop\api\moglix-generative-ai-6dbda5816de0 1.json"
)
vertexai.init(
    project="moglix-generative-ai",
    location="asia-south1",
   
    credentials=credentials
)
# === Get grounded Gemini model + Google Search tool ===
gemini_model, google_tool = get_tools_and_model()

# === Tool 1: Real-time search using Gemini ===
def real_time_google_search(query: str) -> str:
    response = gemini_model.generate_content(query)
    return response.text

google_search_tool = LC_Tool(
    name="google_search",
    func=real_time_google_search,
    description="Use this tool to get real-time info using Google's search engine."
)
# Load both vectorstores
small_vectorstore = load_vector_store("small_vector_db")
full_vectorstore = load_vector_store("policy_vector_db")

# Convert them to retrievers
small_retriever = small_vectorstore.as_retriever(search_kwargs={"k": 3})
full_retriever = full_vectorstore.as_retriever(search_kwargs={"k": 5})


# Create one QA chain but don't bind it to a specific retriever yet
def get_qa_chain(retriever):
    return RetrievalQA.from_chain_type(
        llm=VertexAI(model_name="gemini-1.5-flash"),
        retriever=retriever
    )

def dynamic_internal_policy_tool(query: str) -> str:
    query_vector = embed_query(query)
    small_docs = small_vectorstore.similarity_search_by_vector(query_vector.tolist(), k=3)

    # Manually calculate cosine similarity between query and each doc
    best_score = -1
    best_doc = None
    for doc in small_docs:
        doc_vector = embed_query(doc.page_content)  # embed the document content
        score = cosine_similarity(query_vector, doc_vector)
        if score > best_score:
            best_score = score
            best_doc = doc

    print(f"🔍 Small DB Cosine Score: {best_score:.3f}")

    if best_score >= 0.5:
        print("✅ Answering using small vector DB.")
        return get_qa_chain(small_retriever).run(query)

    print("🔄 Falling back to full vector DB")
    return get_qa_chain(full_retriever).run(query)

access_policy_tool = LC_Tool(
    func=dynamic_internal_policy_tool,
    name="internal_policy_assistant",
    description=(
        "Use this tool to answer questions related to Moglix's internal policies, "
        "including salary advance, medical emergency contact, leaves, HR policies, "
        "access control, user rights, gratuity, or other company-specific topics. "
        "It retrieves accurate answers from company policy PDFs."
    )
)


# === All tools ===
tools = [google_search_tool, access_policy_tool]

# === LLM and Memory ===
llm = VertexAI(model_name="gemini-1.5-flash", temperature=0.4)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

CUSTOM_PROMPT = PromptTemplate.from_template("""
You are an intelligent AI assistant having a conversation with a human.
Use the following conversation history and the current question to provide a relevant and contextual response.
Conversation history:
{chat_history}

Current question:
{input}

Answer as helpfully and concisely as possible.
""")
# === Initialize Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
   agent_kwargs={
    "input_variables": ["input", "chat_history"],
    "prefix": CUSTOM_PROMPT.template
}
)
# === CLI Interface ===
if __name__ == "__main__":
    print("🤖 LangChain Agent with PDF Knowledge + Memory")
    while True:
       try:
         query = input("You: ")
         if query.lower() in ["exit", "quit"]:
                break
         response = check_cache(query)
         if response:
            print("Bot (cached):", response)
         else:
           response = agent.run(query)
           update_cache(query, response)
           print("Bot:", response)
       except KeyboardInterrupt:
            print("Goodbye!")
            break

