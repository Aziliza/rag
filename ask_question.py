import argparse
import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import warnings
warnings.filterwarnings("ignore")

api_base="https://api.chatanywhere.tech/v1"
api_key="sk-6lDdDxc8f9V7eEuT9egSqhe3VbtLyTBLHnr3LP5faS9vSf0s"
embeddings = OpenAIEmbeddings(openai_api_base=api_base,openai_api_key=api_key)
parser = argparse.ArgumentParser(description='Paepper.com Q&A')
parser.add_argument('question', type=str, help='Your question for Paepper.com')
args = parser.parse_args()

store=FAISS.load_local("faiss_store",embeddings,allow_dangerous_deserialization=True)


os.environ["OPENAI_API_KEY"]=api_key
os.environ["OPENAI_API_BASE"]=api_base
# 由于某些原因，使用OpenAI会报错，改为ChatOpenAI。来源：https://github.com/langchain-ai/langchain/issues/1643
llm=ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0, verbose=True,openai_api_base=api_base,openai_api_key=api_key)
chain = VectorDBQAWithSourcesChain.from_llm(
        llm=llm, vectorstore=store, verbose=True)
result = chain({"question": args.question})

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")