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
너는 '주식 관련 키워드 검색'을 위한 쿼리 확장기야.
입력 키워드를 주식/산업 카테고리 관점에서 더 구체화해 1~2문장으로 확장해.
- 동의어/상위카테고리를 포함해.
- 너무 길게 쓰지 말고, 검색에 도움 되는 단어 위주로.
- 절대 특정 종목을 직접 언급하거나 추천하지 마.

키워드: {keyword}
확장쿼리:
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
