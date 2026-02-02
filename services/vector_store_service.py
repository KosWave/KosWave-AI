"""Vector store service for managing ChromaDB"""
import json
import os
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import Config


class VectorStoreService:
    """ChromaDB 벡터 스토어 관리 서비스 (Singleton)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize vector store (only once)"""
        if VectorStoreService._initialized:
            return
            
        print("🔧 Vector Store 초기화 중...")
        
        # Embeddings 초기화
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # ChromaDB 벡터 스토어 초기화
        self.vectorstore = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # 데이터가 없으면 로드
        if self.vectorstore._collection.count() == 0:
            print("📊 주식 데이터 로딩 중...")
            self._load_stock_data()
        else:
            print(f"✅ 기존 Vector DB 로드 완료 (총 {self.vectorstore._collection.count()}개 종목)")
        
        VectorStoreService._initialized = True
    
    def _load_stock_data(self):
        """주식 데이터를 로드하여 벡터 DB 구축"""
        stock_data_path = Config.STOCK_DATA_PATH
        
        if not os.path.exists(stock_data_path):
            raise FileNotFoundError(f"Stock data file not found: {stock_data_path}")
        
        with open(stock_data_path, "r", encoding="utf-8") as f:
            stock_texts = json.load(f)
        
        texts = []
        metadatas = []
        
        for item in stock_texts:
            tags = item.get("tags", [])
            
            # 종목 정보를 텍스트로 결합
            combined_content = f"""종목명: {item['name']}
산업: {item['industry']}
설명: {item['description']}
세부내용: {' '.join(item['comments'])}
연관키워드: {', '.join(tags)}
""".strip()
            
            texts.append(combined_content)
            
            metadatas.append({
                "market": item['market'],
                "code": item['code'],
                "name": item['name'],
                "industry": item['industry'],
            })
        
        # 벡터 DB에 추가
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"✅ Vector DB 구축 완료! (총 {len(texts)}개 종목)")
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        유사도 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            
        Returns:
            (Document, distance) 튜플의 리스트
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)
