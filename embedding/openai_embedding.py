import os
from openai import OpenAI
import chromadb
import numpy as np
from dotenv import load_dotenv
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# ChromaDBの保存先ディレクトリ
PERSIST_DIRECTORY = "chroma_db"
# 使用する埋め込みモデル（環境変数から読み込み、デフォルト値を設定）
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
logger.info(f"使用する埋め込みモデル: {EMBEDDING_MODEL}")

def load_api_key():
    """APIキーを環境変数から読み込む"""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEYが設定されていません")
    return api_key

def create_embeddings(text_lines, max_lines=5):
    """テキストのembeddingを生成"""
    embeddings = []
    client = OpenAI(api_key=load_api_key())
    try:
        for i, line in enumerate(text_lines[:max_lines]):
            if not line.strip():
                continue
            response = client.embeddings.create(
                input=line.strip(),
                model=EMBEDDING_MODEL
            )
            embedding_vector = response.data[0].embedding
            embeddings.append((line.strip(), embedding_vector))
            
            # embedding値の情報を出力
            logger.info(f"Line {i+1}:")
            logger.info(f"テキスト: {line.strip()}")
            logger.info(f"Embedding次元数: {len(embedding_vector)}")
            logger.info(f"Embedding値（最初の5次元）: {embedding_vector[:5]}")
            logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"OpenAI API エラー: {str(e)}")
        raise
    return embeddings

def setup_chromadb():
    """ChromaDBのセットアップ"""
    # 保存ディレクトリが存在しない場合は作成
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)

    # ChromaDBクライアントの新しい設定方式
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    return client

def main():
    try:
        # テキストファイルの読み込み
        input_file = 'novel/Ncomic_生贄姫.txt'
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"ファイルが見つかりません: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Embeddingの生成
        embeddings = create_embeddings(lines)

        # ChromaDBへの保存
        client = setup_chromadb()
        try:
            # コレクション名の生成
            collection_name = f"Ncomic_sacrifice_princess_{EMBEDDING_MODEL}"
            
            # 既存のコレクションを削除（存在する場合）
            try:
                client.delete_collection(collection_name)
            except ValueError:
                pass

            collection = client.create_collection(name=collection_name)
            
            for i, (text, embedding) in enumerate(embeddings):
                collection.add(
                    documents=[text],
                    embeddings=[embedding],
                    metadatas=[{"line_number": i + 1}],
                    ids=[f"text_{i+1}"]
                )
            
            logger.info("Embeddingsを ChromaDBに保存しました")

        except Exception as e:
            logger.error(f"ChromaDB エラー: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()
