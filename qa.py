import sys
import boto3
from langchain_aws import BedrockLLM
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def main(query):
    session = boto3.Session()
    bedrock_runtime = session.client(service_name='bedrock-runtime')

    llm = BedrockLLM(
        model_id="anthropic.claude-v2:1",
        client=bedrock_runtime,
        model_kwargs={"max_tokens_to_sample": 16000},
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime
    )

    vectorstore = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    prompt_template = """与えられた文脈情報のみを使用して、以下の質問に答えてください。
    文脈情報に関連する内容がない場合は、「申し訳ありませんが、この質問に答えるための情報が文書にありません。」と回答してください。
    推測や一般的な知識に基づく回答は避け、与えられた情報にのみ基づいて回答してください。

    文脈情報:
    {context}

    質問: {question}

    回答: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = chain({"query": query})

    print(f"質問: {query}")
    print(f"回答: {result['result']}")
    print("\n参照された文書:")
    for doc in result['source_documents']:
        print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python qa.py '質問文'")
        sys.exit(1)
    query = sys.argv[1]
    main(query)