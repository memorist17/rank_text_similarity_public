import os
from openai import OpenAI
import chromadb
import numpy as np
from dotenv import load_dotenv
import logging
import hashlib

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

def create_embeddings(text):
    """テキストのembeddingを生成"""
    client = OpenAI(api_key=load_api_key())
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def calculate_similarity(embedding1, embedding2):
    """2つのベクターの類似度を計算"""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def get_text_hash(text):
    """テキストのハッシュ値を生成"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_collection_name(file_path):
    """ファイルパスからコレクション名を生成（日本語文字を英数字に変換）"""
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # 日本語文字を含むファイル名をハッシュ化して英数字のみの文字列に変換
    hashed_name = hashlib.md5(base_name.encode('utf-8')).hexdigest()[:8]
    return f"collection_{hashed_name}_{EMBEDDING_MODEL}"

def process_file(input_file, client):
    """ファイルを処理し、行ごとにembeddingを生成"""
    logger.info(f"処理開始: {input_file}")
    if not os.path.exists(input_file):
        logger.error(f"ファイルが見つかりません: {input_file}")
        return None

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]  # 空行を除外

    collection_name = get_collection_name(input_file)
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)
        logger.info("新しいコレクションを作成しました")

    embeddings = []
    documents = []
    metadatas = []
    ids = []

    for i, line in enumerate(lines):
        text_hash = get_text_hash(f"{line}_{i}")  # インデックスを含めてユニークなハッシュを生成
        
        try:
            # 既存のembeddingをチェック
            existing_docs = collection.get(
                where={"hash": text_hash},
                include=['embeddings']
            )
            
            if existing_docs['ids']:
                logger.info(f"行 {i+1}: 既存のembeddingが見つかりました。スキップします。")
                continue
            
            # 新しいembeddingを生成
            logger.info(f"行 {i+1}: 新しいembeddingを生成します。")
            embedding = create_embeddings(line)
            
            embeddings.append(embedding)
            documents.append(line)
            metadatas.append({"hash": text_hash, "line_number": i+1})
            ids.append(text_hash)
            
        except Exception as e:
            logger.error(f"行 {i+1} の処理中にエラーが発生しました: {str(e)}")
            continue

    # 一括で保存
    if embeddings:
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"{len(embeddings)}行のembeddingを保存しました")

    logger.info(f"処理完了: {input_file}")
    logger.info("-" * 50)
    return collection

def main():
    try:
        # ChromaDBクライアントのセットアップ
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

        # 処理対象のファイルリスト
        input_files = [
            'novel/Ncomic_生贄姫.txt',
            'document/all_simple.txt'
        ]

        # 各ファイルを処理
        collections = {}
        for input_file in input_files:
            collection = process_file(input_file, client)
            if collection:
                collections[input_file] = collection

        # 類似度比較の処理
        logger.info("類似度比較を開始します")
        try:
            # 比較元のコレクション（生贄姫）
            source_collection = collections.get('novel/Ncomic_生贄姫.txt')
            if not source_collection:
                source_collection_name = get_collection_name('novel/Ncomic_生贄姫.txt')
                source_collection = client.get_collection(name=source_collection_name)
            
            source_results = source_collection.get(include=['embeddings'])
            
            if len(source_results['embeddings']) == 0:
                raise ValueError("比較元のembeddingが見つかりません")
            
            source_embedding = source_results['embeddings'][0]

            # 比較先のコレクション（all_simple）
            target_collection = collections.get('document/all_simple.txt')
            if not target_collection:
                target_collection_name = get_collection_name('document/all_simple.txt')
                target_collection = client.get_collection(name=target_collection_name)
            
            target_results = target_collection.get(
                include=['embeddings', 'documents', 'metadatas']
            )

            if len(target_results['embeddings']) == 0:
                raise ValueError("比較先のembeddingが見つかりません")

            # 類似度の計算
            similarities = []
            for i, vector in enumerate(target_results['embeddings']):
                similarity = calculate_similarity(source_embedding, vector)
                line_number = target_results['metadatas'][i].get('line_number', 'Unknown')
                similarities.append((target_results['documents'][i], similarity, line_number))

            # 類似度を降順でソート
            similarities.sort(key=lambda x: x[1], reverse=True)

            # すべての類似度を出力
            logger.info("=== 類似度ランキング ===")
            for i, (text, similarity, line_number) in enumerate(similarities, 1):
                logger.info(f"{i}位: 類似度 {similarity:.4f} (行番号: {line_number})")
                logger.info(f"テキスト: {text}")
                logger.info("-" * 50)

        except Exception as e:
            logger.error(f"類似度比較中にエラーが発生しました: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()
