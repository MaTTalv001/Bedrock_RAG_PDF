#!/bin/bash

if [ $# -eq 0 ]; then
    echo "使用方法: ./query.sh '質問文'"
    exit 1
fi

QUERY="$1"

docker-compose exec bedrock-doc-search python qa.py "$QUERY"