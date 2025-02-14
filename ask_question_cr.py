from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import warnings
import argparse

warnings.filterwarnings("ignore")

api_base = "https://api.chatanywhere.tech/v1"
api_key = "sk-6lDdDxc8f9V7eEuT9egSqhe3VbtLyTBLHnr3LP5faS9vSf0s"

# 加载保存的向量数据库
store = FAISS.load_local(
    "faiss_store",
    embeddings=OpenAIEmbeddings(openai_api_base=api_base, openai_api_key=api_key),
    allow_dangerous_deserialization=True,
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    verbose=True,
    openai_api_base=api_base,
    openai_api_key=api_key,
)
# 返回source documents设置return_source_documents为True就行了
qa = ConversationalRetrievalChain.from_llm(
    llm, store.as_retriever(), return_source_documents=True
)
parser = argparse.ArgumentParser(description="Paepper.com Q&A")
parser.add_argument("question", type=str, help="Your question for Paepper.com")
args = parser.parse_args()
query = args.question
# 不需要chat_history直接设置为空列表就行了
result = qa({"question": query, "chat_history": []})
print(result["answer"])
print("Source:\n", result["source_documents"][0].metadata["source"])
