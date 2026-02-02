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
- 출력은 반드시 아래 JSON 스키마를 따를 것.

{format_instructions}

키워드: {keyword}

후보 종목들:
{candidates}
""")
        
        self.rerank_chain = (
            self.rerank_prompt.partial(format_instructions=self.rerank_parser.get_format_instructions())
            | self.llm
            | self.rerank_parser
        )
        
        # Final Recommendation Chain
        self.final_prompt = ChatPromptTemplate.from_template("""
당신은 금융 전문가입니다.
입력 키워드와 관련주 후보 리스트를 바탕으로 최종 추천 종목을 선정하고 JSON 배열로만 출력하세요.

규칙:
- 반드시 후보 리스트에 있는 종목만 선택
- description은 "후보 텍스트에 존재하는 근거(evidence)"를 기반으로 주어진 **키워드**와의 **연관성**을 제시해야 함
- 억지 연결 금지
- similarity는 0.0~1.0 사이 값 (점수가 높을수록 1.0에 가깝게)
- 최대 {max_results}개까지 선정
- 출력은 JSON 배열 형식
- 출력하는 JSON 배열의 관련주 후보는 10개로 고정.

키워드: {keyword}

재랭킹 결과(점수 높은 순 참고):
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
    
    def get_recommendations(self, keyword: str) -> List[Dict]:
        """
        키워드로 주식 추천
        
        Args:
            keyword: 검색 키워드
            
        Returns:
            추천 종목 리스트 (배열 형태)
        """
        print(f"🔍 keyword = '{keyword}'")
        
        # 1. Query Expansion
        expanded_query = self.query_expander.expand(keyword)
        print(f"🧠 expanded_query = {expanded_query}\n")
        
        # 2. Retrieval (Recall)
        docs_with_scores = self.vector_store.similarity_search_with_score(
            expanded_query, 
            k=Config.RECALL_K
        )
        
        # 디버깅 로그
        print("📋 Recall Top 10 (distance 낮을수록 유사):")
        for i, (doc, distance) in enumerate(docs_with_scores[:10], 1):
            m = doc.metadata
            print(f"  {i:02d}. dist={distance:.4f} | {m['name']}({m['code']}) | {m['industry']}")
        print()
        
        # 3. Reranking
        candidates_text = self._format_candidates_for_rerank(docs_with_scores)
        # print("candidates_text", candidates_text)
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
        
        # 4. Final Recommendation
        reranked_for_final = json.dumps(
            {"items": reranked_items[:Config.RERANK_TOP_K]}, 
            ensure_ascii=False
        )
        
        final_result = self.final_chain.invoke({
            "keyword": keyword,
            "reranked": reranked_for_final,
            "max_results": Config.MAX_SEARCH_RESULTS
        })
        
        # 결과 출력
        print("✨ 추천 결과:\n")
        for idx, rec in enumerate(final_result, 1):
            print(f"🏆 {idx}위: {rec['name']} ({rec['code']})")
            print(f"   이유: {rec['description']}")
            print(f"   유사도: {rec['similarity']:.2f}\n")
        
        # 배열 형태로 반환
        return final_result
    
    def _format_candidates_for_rerank(self, docs_with_scores) -> str:
        """
        Rerank를 위한 후보 종목 포맷팅
        
        Args:
            docs_with_scores: (Document, distance) 튜플 리스트
            
        Returns:
            포맷된 후보 종목 텍스트
        """
        parts = []
        for idx, (doc, distance) in enumerate(docs_with_scores, start=1):
            m = doc.metadata
            # 토큰 폭주 방지를 위해 앞부분만 사용
            content = doc.page_content[:800]
            parts.append(
                f"[{idx}] {m['name']}({m['code']}) | 산업: {m['industry']}\n{content}"
            )
        return "\n\n".join(parts)
