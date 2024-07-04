import os
import boto3
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

def vectorstore_exists():
    return os.path.exists("./vectorstore/index.faiss") and os.path.exists("./vectorstore/index.pkl")

def create_vectorstore():
    print("Starting vectorstore creation process...")
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
    print("Saving vectorstore...")
    vectorstore.save_local("./vectorstore")
    print("Vector store created and saved.")

if __name__ == "__main__":
    if not vectorstore_exists():
        print("Vectorstore not found or incomplete. Creating new one...")
        create_vectorstore()
    else:
        print("Vectorstore already exists. Skipping creation.")
        # 既存のベクトルストアをロードしてテストする
        try:
            embeddings = BedrockEmbeddings(
                model_id="amazon.titan-embed-text-v1",
                client=boto3.Session().client(service_name='bedrock-runtime')
            )
            LangchainFAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)
            print("Existing vectorstore loaded successfully.")
        except Exception as e:
            print(f"Error loading existing vectorstore: {e}")
            print("Recreating vectorstore...")
            create_vectorstore()