"""Stock recommendation service with LLM-based reranking"""
import json
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models.schemas import RerankItem, RerankResult, StockRecommendation
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
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # Rerank Chain
        self.rerank_parser = JsonOutputParser(pydantic_object=RerankResult)
        self.rerank_prompt = ChatPromptTemplate.from_template("""
너는 '관련주 후보 재랭커(reranker)'야.
입력 키워드와 후보 종목들의 설명을 보고, 각 후보가 키워드와 얼마나 직접적으로 관련 있는지 점수화해.

규칙:
- 점수는 0~100.
- "억지 연결"은 점수를 낮게.
- evidence는 반드시 주어진 종목 설명에 근거하여 2~3줄 내외로 간결히 작성할 것.
- 모르면 낮게(0~30) 줘.

{format_instructions}

키워드: {keyword}

후보:
{candidates}
""")
        
        self.rerank_chain = (
            self.rerank_prompt.partial(format_instructions=self.rerank_parser.get_format_instructions())
            | self.llm
            | self.rerank_parser
        )
        
        # Final Recommendation Chain
        self.final_prompt = ChatPromptTemplate.from_template("""
금융 전문가로서 키워드 관련 종목 10개를 선정하고 JSON 배열로만 출력.

규칙:
- 후보 리스트에 있는 종목만 선택
- description은 evidence 기반으로 키워드 연관성 제시 (2줄 이내)
- similarity는 0.0~1.0 (점수 높을수록 1.0)
- 정확히 10개 선정

키워드: {keyword}

후보 (점수순):
{reranked}

출력 예시:
[
  {{
    "name": "삼성전자",
    "code": "005930",
    "description": "반도체 업계를 선도하는 기업으로, DRAM과 낸드 메모리에서 세계 1위를 차지하고 있습니다.",
    "similarity": 0.97
  }},
  {{
    "name": "SK하이닉스",
    "code": "000660",
    "description": "메모리 반도체 분야의 글로벌 리더로, DRAM 및 낸드 플래시 생산에 주력하고 있습니다.",
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
            k=Config.RECALL_K
        )
        
        # 2-2. 뉴스 데이터 Retrieval (키워드 기반)
        news_docs_with_scores = self.vector_store.search_news_by_keyword(
            expanded_query,
            k=30  # 뉴스는 더 많이 검색하여 다양한 종목 커버
        )
        
        # 뉴스에서 추출한 종목 코드 집합
        news_stock_codes = set()
        for doc, _ in news_docs_with_scores:
            news_stock_codes.add(doc.metadata.get('code'))
        
        print(f"📰 뉴스에서 발견된 종목 수: {len(news_stock_codes)}")
        
        # 디버깅 로그
        print("📋 Recall Top 10 (distance 낮을수록 유사):")
        for i, (doc, distance) in enumerate(docs_with_scores[:10], 1):
            m = doc.metadata
            in_news = "📰" if m['code'] in news_stock_codes else "  "
            print(f"  {i:02d}. dist={distance:.4f} | {m['name']}({m['code']}) | {m['industry']} {in_news}")
        print()
        
        # 3. Reranking (뉴스 정보를 컨텍스트에 추가)
        candidates_text = self._format_candidates_for_rerank(docs_with_scores, news_docs_with_scores)
        reranked = self.rerank_chain.invoke({
            "keyword": keyword,
            "candidates": candidates_text
        })
        
        # 점수순 정렬
        reranked_items = sorted(reranked["items"], key=lambda x: x["score"], reverse=True)
        
        print("🏁 Rerank Top 10:")
        for i, item in enumerate(reranked_items[:10], 1):
            print(f"  {i:02d}. score={item['score']:3d} | {item['stockName']}({item['stockCode']}) | evidence={item['evidence']}")
        print()
        
        # 4. Final Recommendation (뉴스 없이 빠르게 처리)
        reranked_for_final = json.dumps(
            {"items": reranked_items[:Config.RERANK_TOP_K]}, 
            ensure_ascii=False
        )
        
        print("🤖 LLM 최종 추천 생성 중...")
        final_result = self.final_chain.invoke({
            "keyword": keyword,
            "reranked": reranked_for_final,
            "max_results": Config.MAX_SEARCH_RESULTS
        })
        
        # 5. 결과 검증 및 뉴스 추가
        print("📰 관련 뉴스 검색 중...")
        valid_results = []
        for rec in final_result:
            # 필수 필드 검증
            if not all(key in rec for key in ['name', 'code', 'description', 'similarity']):
                print(f"⚠️ 잘못된 응답 형식, 건너뜀: {rec}")
                continue
            
            stock_code = rec['code']
            news_list = self.vector_store.search_news_by_stock_code(stock_code, k=3)
            rec['news'] = news_list
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
                # 뉴스 제목만 저장 (최대 3개)
                if len(stock_news_map[code]) < 3:
                    stock_news_map[code].append(title)
        
        parts = []
        for idx, (doc, distance) in enumerate(docs_with_scores, start=1):
            m = doc.metadata
            # 토큰 폭주 방지를 위해 앞부분만 사용 (400자로 축소)
            content = doc.page_content[:400]
            
            # 뉴스 정보 추가
            news_section = ""
            if m['code'] in stock_news_map:
                news_titles = stock_news_map[m['code']]
                news_section = "\n관련 뉴스:\n" + "\n".join([f"- {title}" for title in news_titles])
            
            parts.append(
                f"[{idx}] {m['name']}({m['code']}) | 산업: {m['industry']}\n{content}{news_section}"
            )
        return "\n\n".join(parts)
