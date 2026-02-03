"""Query expansion using LLM for keywords without rule-based synonyms"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .keyword_synonyms import rule_expand_keyword
from config import Config


class QueryExpander:
    """쿼리 확장기: 룰 기반 우선, LLM fallback"""
    
    def __init__(self):
        """Initialize LLM-based query expander"""
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        self.expand_prompt = ChatPromptTemplate.from_template("""
너는 '주식 검색 최적화'를 수행하는 쿼리 확장기야.
입력된 키워드가 속한 **산업/분야**를 먼저 명확히 하고, 그 안에서 관련된 주식 테마 키워드를 나열해.
동음이의어(예: 배 vs 배, 머플러 vs 머플러)일 가능성이 있다면 **주요한 2가지 의미**를 각각 "의미1 / 의미2" 형태로 나누어 확장해.
토큰 input이 지나치지 않게 단어는 최대 10개까지 간결하게 확장해.

형식:
[상위 카테고리] 관련 키워드: 핵심단어1, 핵심단어2, ...

예시 1 (단일 의미):
입력: 불닭볶음면
확장: [식품/라면] 관련 키워드: 매운라면, 인스턴트 식품, 삼양식품, 수출 실적, K-푸드, 편의점

예시 2 (중의적):
입력: 머플러
확장: [패션/의류] 관련 키워드: 방한용품, 겨울 의류, 스카프, 섬유 제조, 백화점 / [자동차 부품] 관련 키워드: 배기 시스템, 소음기, 자동차 매연 저감장치, 친환경 부품

입력: {keyword}
확장:
""")
        
        self.query_expander_chain = self.expand_prompt | self.llm | StrOutputParser()
    
    def expand(self, keyword: str) -> str:
        """
        키워드 확장: 룰 기반 우선, 없으면 LLM 확장
        
        Args:
            keyword: 입력 키워드
            
        Returns:
            확장된 쿼리 문자열
        """
        # 1. 룰 기반 확장 우선
        rule_result = rule_expand_keyword(keyword)
        if rule_result:
            return rule_result
        
        # 2. LLM 기반 확장 (fallback)
        return self.query_expander_chain.invoke({"keyword": keyword}).strip()
