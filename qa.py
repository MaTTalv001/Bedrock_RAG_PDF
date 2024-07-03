from langchain.llms import Bedrock
from langchain.embeddings import BedrockEmbeddings

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

llm = Bedrock(
    model_id="anthropic.claude-v2",
    model_kwargs={"max_tokens_to_sample": 16000},
)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

vectorstore = FAISS.load_local("./vectorstore", embeddings)
retriever = vectorstore.as_retriever()

chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

results = chain("Amazon Kendraの仕組みについて教えてください。")

print(results["result"])

