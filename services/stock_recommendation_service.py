"""Stock recommendation service with LLM-based reranking"""
import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models.schemas import StockRecommendation
from utils.query_expander import QueryExpander
from services.vector_store_service import VectorStoreService
from config import Config


class StockRecommendationService:
    """주식 추천 서비스: 쿼리 확장 → Retrieval → Rerank → Final"""
    
    def __init__(self):
        """Initialize recommendation service"""
        self.vector_store = VectorStoreService()
        self.query_expander = QueryExpander()
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=Config.FAST_LLM_MODEL if Config.FAST_MODE else Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # One-Shot Recommendation Chain (Selection + Explanation)
        self.final_prompt = ChatPromptTemplate.from_template("""
키워드: {keyword}

관련주 후보:
{candidates}

지시사항:
1. 위 후보 중 키워드와 가장 연관성 높은 6개 종목을 선정하세요.
2. 각 종목 선정 이유(description)를 1문장으로 핵심만 간단하게 요약하세요.
3. 연관성 점수(similarity)는 0.0~1.0입니다.
4. 아래 JSON 포맷으로 출력하세요. (JSON만 출력)

[
  {{
    "name": "종목명",
    "code": "종목코드",
    "description": "핵심 선정 이유",
    "similarity": 0.95
  }}
]
""")
        
        # StrOutputParser 대신 JsonOutputParser 사용
        self.final_parser = JsonOutputParser()
        self.final_chain = self.final_prompt | self.llm | self.final_parser
    
    def get_recommendations(self, keyword: str):
        """
        키워드로 주식 추천 (뉴스 데이터 통합)
        
        Args:
            keyword: 검색 키워드
            
        Returns:
            추천 종목 리스트 (배열 형태, 뉴스 포함)
        """
        print(f"🔍 keyword = '{keyword}'")
        
        # 1. Query Expansion
        expanded_query = self.query_expander.expand(keyword)
        print(f"🧠 expanded_query = {expanded_query}\n")
        
        # 2-1. 주식 정보 Retrieval (Recall)
        docs_with_scores = self.vector_store.similarity_search_with_score(
            expanded_query, 
            k=Config.FAST_RECALL_K if Config.FAST_MODE else Config.RECALL_K
        )
        
        # 2-2. 뉴스 데이터 Retrieval (키워드 기반)
        news_docs_with_scores = []
        if Config.FAST_MODE:
            if Config.FAST_NEWS_K > 0:
                news_docs_with_scores = self.vector_store.search_news_by_keyword(
                    expanded_query,
                    k=Config.FAST_NEWS_K
                )
        else:
            news_docs_with_scores = self.vector_store.search_news_by_keyword(
                expanded_query,
                k=30  # 뉴스는 더 많이 검색하여 다양한 종목 커버
            )
        
        # 뉴스 from extraction (재활용을 위해 딕셔너리 저장)
        stock_news_map = {}
        for doc, _ in news_docs_with_scores:
            code = doc.metadata.get('code')
            news_item = {
                "title": doc.metadata.get('title', ''),
                "link": doc.metadata.get('link', ''),
                "published_date": doc.metadata.get('published_date', '')
            }
            if code not in stock_news_map:
                stock_news_map[code] = []
            # 뉴스 최대 3개 저장 (최종 출력용)
            if len(stock_news_map[code]) < 3:
                stock_news_map[code].append(news_item)
                
        # 뉴스에서 추출한 종목 코드 집합
        news_stock_codes = set(stock_news_map.keys())
        
        print(f"📰 뉴스에서 발견된 종목 수: {len(news_stock_codes)}")
        
        # 디버깅 로그
        print("📋 Recall Top 10 (distance 낮을수록 유사):")
        for i, (doc, distance) in enumerate(docs_with_scores[:10], 1):
            m = doc.metadata
            in_news = "📰" if m['code'] in news_stock_codes else "  "
            print(f"  {i:02d}. dist={distance:.4f} | {m['name']}({m['code']}) | {m['industry']} {in_news}")
        print()
        
        # 3. One-Shot Selection & Explanation
        # Rerank 단계 없이 바로 후보군을 포맷팅하여 최종 추천 프롬프트에 넘김.
        candidates_text = self._format_candidates_for_rerank(docs_with_scores, news_docs_with_scores)
        
        print("🤖 LLM 최종 추천 생성 중... (One-Shot)")
        final_result = self.final_chain.invoke({
            "keyword": keyword,
            "candidates": candidates_text
        })
        
        # 5. 결과 검증 및 뉴스 추가 (DB 재조회 없이 매핑된 뉴스 사용)
        print("📰 관련 뉴스 매핑 중... (DB 재조회 X)")
        valid_results = []
        for rec in final_result:
            # 필수 필드 검증
            if not all(key in rec for key in ['name', 'code', 'description', 'similarity']):
                print(f"⚠️ 잘못된 응답 형식, 건너뜀: {rec}")
                continue
            
            stock_code = rec['code']
            # 기존 뉴스 매핑 활용 (속도 최적화)
            rec['news'] = stock_news_map.get(stock_code, [])
            
            # Fallback: 매핑된 뉴스가 없으면 DB 조회 (정확도 보장)
            if not rec['news']:
                # print(f"⚠️ 뉴스 보완 검색: {stock_code}")
                rec['news'] = self.vector_store.search_news_by_stock_code(stock_code, k=3)
            
            valid_results.append(rec)
        
        # 결과 출력
        print("✨ 추천 결과:\n")
        for idx, rec in enumerate(valid_results, 1):
            print(f"🏆 {idx}위: {rec['name']} ({rec['code']})")
            print(f"   이유: {rec['description']}")
            print(f"   유사도: {rec['similarity']:.2f}")
            print(f"   관련 뉴스: {len(rec['news'])}건\n")
        
        # 배열 형태로 반환
        return valid_results
    
    def _format_candidates_for_rerank(self, docs_with_scores, news_docs_with_scores=None) -> str:
        """
        Rerank를 위한 후보 종목 포맷팅 (뉴스 정보 포함)
        
        Args:
            docs_with_scores: (Document, distance) 튜플 리스트 (주식 정보)
            news_docs_with_scores: (Document, distance) 튜플 리스트 (뉴스 정보)
            
        Returns:
            포맷된 후보 종목 텍스트
        """
        # 종목별 뉴스 매핑
        stock_news_map = {}
        if news_docs_with_scores:
            for doc, _ in news_docs_with_scores:
                code = doc.metadata.get('code')
                title = doc.metadata.get('title', '')
                if code not in stock_news_map:
                    stock_news_map[code] = []
                if len(stock_news_map[code]) < 1:  # 뉴스 1개만 포함 (토큰 절약)
                    stock_news_map[code].append(title)
        
        parts = []
        for idx, (doc, distance) in enumerate(docs_with_scores, start=1):
            m = doc.metadata
            # 토큰 폭주 방지를 위해 앞부분만 사용
            content_limit = 120 if Config.FAST_MODE else 200
            content = doc.page_content[:content_limit]
            
            # 뉴스 정보 추가
            news_section = ""
            if m['code'] in stock_news_map:
                news_titles = stock_news_map[m['code']]
                # 뉴스 제목만 한줄로 추가
                news_section = f" | 뉴스: {news_titles[0]}"
            
            # 중복 제거 (page_content에 이미 종목명/산업이 있으므로 앞부분 헤더 제거)
            # content 형식이 "종목명: ... 산업: ..." 이므로 그대로 둠
            parts.append(
                f"[{idx}] {content} (Code: {m['code']}){news_section}"
            )
        return "\n\n".join(parts)
