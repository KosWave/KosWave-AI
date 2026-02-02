"""Vector store service for managing ChromaDB"""
import json
import os
from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import Config
from typing import List, Dict


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
        
        # ChromaDB 벡터 스토어 초기화 (주식 정보)
        self.vectorstore = Chroma(
            collection_name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # 뉴스 벡터 스토어 초기화
        self.news_vectorstore = Chroma(
            collection_name=Config.CHROMA_NEWS_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.CHROMA_DB_PATH,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # 주식 데이터가 없으면 로드
        if self.vectorstore._collection.count() == 0:
            print("📊 주식 데이터 로딩 중...")
            self._load_stock_data()
        else:
            print(f"✅ 기존 주식 Vector DB 로드 완료 (총 {self.vectorstore._collection.count()}개 종목)")
        
        # 뉴스 데이터가 없으면 로드
        if self.news_vectorstore._collection.count() == 0:
            print("📰 뉴스 데이터 로딩 중...")
            self._load_news_data()
        else:
            print(f"✅ 기존 뉴스 Vector DB 로드 완료 (총 {self.news_vectorstore._collection.count()}개 뉴스)")
        
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
    
    def _load_news_data(self):
        """뉴스 데이터를 로드하여 벡터 DB 구축 (배치 처리)"""
        import time
        
        news_data_path = Config.NEWS_DATA_PATH
        
        if not os.path.exists(news_data_path):
            print(f"⚠️ 뉴스 데이터 파일 없음: {news_data_path}")
            return
        
        with open(news_data_path, "r", encoding="utf-8") as f:
            news_list = json.load(f)
        
        print(f"📰 총 {len(news_list)}개 뉴스 로딩 시작 (배치 처리)...")
        
        # 배치 크기 설정 (rate limit 고려)
        batch_size = 100
        total_batches = (len(news_list) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(news_list), batch_size):
            batch_news = news_list[batch_idx:batch_idx + batch_size]
            
            texts = []
            metadatas = []
            
            for item in batch_news:
                # 뉴스 제목과 내용을 결합하여 임베딩
                # content는 100자로 제한하여 토큰 절약
                content_preview = item['content'][:100] + "..." if len(item['content']) > 100 else item['content']
                
                combined_content = f"""제목: {item['title']}
내용: {content_preview}
종목: {item['name']}
""".strip()
                
                texts.append(combined_content)
                
                metadatas.append({
                    "code": item['code'],
                    "name": item['name'],
                    "title": item['title'],
                    "content": content_preview,
                    "link": item['link'],
                    "published_date": item['published_date']
                })
            
            # 벡터 DB에 추가
            current_batch = batch_idx // batch_size + 1
            print(f"   배치 {current_batch}/{total_batches} 처리 중... ({len(texts)}개 뉴스)")
            
            try:
                self.news_vectorstore.add_texts(texts=texts, metadatas=metadatas)
                
                # Rate limit 방지를 위해 대기 (마지막 배치는 제외)
                if batch_idx + batch_size < len(news_list):
                    time.sleep(2)
                    
            except Exception as e:
                print(f"   ⚠️ 배치 {current_batch} 처리 중 오류: {e}")
                print(f"   20초 대기 후 재시도...")
                time.sleep(20)
                # 재시도
                try:
                    self.news_vectorstore.add_texts(texts=texts, metadatas=metadatas)
                except Exception as retry_error:
                    print(f"   ❌ 재시도 실패: {retry_error}")
                    print(f"   배치 {current_batch} 건너뜀")
                    continue
        
        total_loaded = self.news_vectorstore._collection.count()
        print(f"✅ 뉴스 Vector DB 구축 완료! (총 {total_loaded}개 뉴스)")
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        유사도 검색 수행 (주식 정보)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 개수
            
        Returns:
            (Document, distance) 튜플의 리스트
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def search_news_by_keyword(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        키워드로 뉴스 검색
        
        Args:
            query: 검색 키워드
            k: 반환할 뉴스 개수
            
        Returns:
            (Document, distance) 튜플의 리스트
        """
        return self.news_vectorstore.similarity_search_with_score(query, k=k)
    
    def search_news_by_stock_code(
        self, 
        stock_code: str, 
        k: int = 5
    ) -> List[Dict]:
        """
        종목 코드로 관련 뉴스 검색
        
        Args:
            stock_code: 종목 코드
            k: 반환할 뉴스 개수
            
        Returns:
            뉴스 리스트 (메타데이터)
        """
        # ChromaDB의 where 필터 사용
        results = self.news_vectorstore._collection.get(
            where={"code": stock_code},
            limit=k
        )
        
        # 메타데이터만 추출 (content는 이미 100자로 제한됨)
        news_list = []
        if results and results.get('metadatas'):
            for metadata in results['metadatas']:
                news_list.append({
                    "title": metadata.get('title', ''),
                    "content": metadata.get('content', ''),
                    "link": metadata.get('link', ''),
                    "published_date": metadata.get('published_date', '')
                })
        
        return news_list
