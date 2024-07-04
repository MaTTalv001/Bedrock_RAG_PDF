import os
import boto3
from botocore.exceptions import ClientError
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

try:
    # AWS Bedrockサービスへの接続を設定
    session = boto3.Session()
    bedrock_runtime = session.client(service_name='bedrock-runtime')

    # Bedrockの埋め込みモデルを初期化
    # これにより、テキストをベクトルに変換できるようになる
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime
    )

    # テスト用の埋め込みを実行
    # これにより、埋め込みモデルが正常に機能していることを確認
    # test_embedding = embeddings.embed_query("Test query")
    # print("Test embedding successful:", len(test_embedding))

    # FAISSベクトルストアの初期化
    # ダミーデータを使用して初期化し、すぐに削除する
    # これにより、空のベクトルストアが作成される
    dummy_text, dummy_id = "1", 1
    vectorstore = FAISS.from_texts([dummy_text], embeddings, ids=[dummy_id])
    vectorstore.delete([dummy_id])

    # 処理対象のPDFファイルを含むディレクトリを指定
    dirname = "datasets"
    files = []

    # ディレクトリ内のファイルをリストアップ
    for filename in os.listdir(dirname):
        full_path = os.path.join(dirname, filename)
        if os.path.isfile(full_path):
            files.append({"name": filename, "path": full_path})

    # 各ファイルを処理
    for file in files:
        if file["path"].endswith(".pdf"):
            # PDFローダーを使用してファイルを読み込む
            loader = PyPDFLoader(file["path"])
        else:
            # PDFファイル以外はエラーを発生させる
            raise ValueError(f"Unsupported file format: {file['path']}")

        # PDFをページ単位で読み込み、分割
        pages = loader.load_and_split()
        for page in pages:
            # 各ページにメタデータ（ソースファイル情報）を追加
            page.metadata["source"] = file["path"]
            page.metadata["name"] = file["name"]

        # テキストを小さなチャンクに分割
        # これにより、長い文書を管理可能なサイズに分割できる
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)

        # 分割されたドキュメントをベクトル化し、FAISSインデックスに追加
        # これにより、テキストがベクトルに変換され、効率的に検索可能になる
        vectorstore.merge_from(FAISS.from_documents(docs, embeddings))

    # 作成されたベクトルストアをローカルに保存
    # これにより、後で再利用することができる
    vectorstore.save_local("./vectorstore")
    print("Vectorstore saved successfully")

except ClientError as e:
    # AWS固有のエラーをキャッチして詳細を表示
    print(f"AWS Client Error: {e}")
    print(f"Error Code: {e.response['Error']['Code']}")
    print(f"Error Message: {e.response['Error']['Message']}")
except Exception as e:
    # その他の予期しないエラーをキャッチ
    print(f"Unexpected error: {e}")