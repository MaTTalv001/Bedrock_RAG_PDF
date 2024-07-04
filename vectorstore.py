import os
import boto3
from botocore.exceptions import ClientError
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

try:
    session = boto3.Session()
    bedrock_runtime = session.client(service_name='bedrock-runtime')

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime
    )

    # テスト用の埋め込み
    test_embedding = embeddings.embed_query("Test query")
    print("Test embedding successful:", len(test_embedding))

    # 以下は変更なし
    dummy_text, dummy_id = "1", 1
    vectorstore = FAISS.from_texts([dummy_text], embeddings, ids=[dummy_id])
    vectorstore.delete([dummy_id])

    dirname = "datasets"
    files = []

    for filename in os.listdir(dirname):
        full_path = os.path.join(dirname, filename)
        if os.path.isfile(full_path):
            files.append({"name": filename, "path": full_path})

    for file in files:
        if file["path"].endswith(".pdf"):
            loader = PyPDFLoader(file["path"])
        else:
            raise ValueError(f"Unsupported file format: {file['path']}")

        pages = loader.load_and_split()
        for page in pages:
            page.metadata["source"] = file["path"]
            page.metadata["name"] = file["name"]

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        vectorstore.merge_from(FAISS.from_documents(docs, embeddings))

    vectorstore.save_local("./vectorstore")
    print("Vectorstore saved successfully")

except ClientError as e:
    print(f"AWS Client Error: {e}")
    print(f"Error Code: {e.response['Error']['Code']}")
    print(f"Error Message: {e.response['Error']['Message']}")
except Exception as e:
    print(f"Unexpected error: {e}")