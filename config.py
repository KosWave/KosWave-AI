import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask 설정
    # SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'

    # OpenAI 설정
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Embedding 모델 설정
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    
    # LLM 모델 설정 (temperature 0으로 고정하여 추천 결과 흔들림 방지)
    # LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-5-mini-2025-08-07')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.0'))

    # ChromaDB 설정
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './data/chroma_db')
    CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'stocks')
    CHROMA_NEWS_COLLECTION_NAME = os.getenv('CHROMA_NEWS_COLLECTION_NAME', 'news')

    
    # 주식 데이터 파일 경로
    STOCK_DATA_PATH = os.getenv('STOCK_DATA_PATH', './data/stock_info.json')
    NEWS_DATA_PATH = os.getenv('NEWS_DATA_PATH', './data/news.json')

    # 검색 설정
    MAX_SEARCH_RESULTS = int(os.getenv('MAX_SEARCH_RESULTS', 10))
    
    # Retrieval 설정 (Recall을 크게 해서 후보군 누락 방지)
    RECALL_K = int(os.getenv('RECALL_K', '20'))
    RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', '12'))

    @staticmethod
    def validate():
        """필수 설정 검증"""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env file")

        # ChromaDB 디렉토리 생성
        os.makedirs(Config.CHROMA_DB_PATH, exist_ok=True)