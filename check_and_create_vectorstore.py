import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import boto3

def create_vectorstore():
    print("Creating new vector store...")
    session = boto3.Session()
    bedrock_runtime = session.client(service_name='bedrock-runtime')

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime
    )

    dirname = "datasets"
    files = []

    for filename in os.listdir(dirname):
        full_path = os.path.join(dirname, filename)
        if os.path.isfile(full_path) and filename.endswith('.pdf'):
            files.append({"name": filename, "path": full_path})

    documents = []
    for file in files:
        loader = PyPDFLoader(file["path"])
        pages = loader.load_and_split()
        for page in pages:
            page.metadata["source"] = file["path"]
            page.metadata["name"] = file["name"]
        documents.extend(pages)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("./vectorstore")
    print("Vector store created and saved.")

if __name__ == "__main__":
    if not os.path.exists("./vectorstore"):
        create_vectorstore()
    else:
        print("Vector store already exists. Skipping creation.")