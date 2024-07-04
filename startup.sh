#!/bin/bash

# ベクトルストアの確認と作成
python check_and_create_vectorstore.py > vectorstore_creation.log 2>&1

# ログの内容を表示
cat vectorstore_creation.log

# スタートアップメッセージ
echo "コンテナが起動しました。クエリを実行するには外部からquery.shを使用してください。"