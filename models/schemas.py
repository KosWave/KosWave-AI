"""Pydantic schemas for stock recommendation system"""
from pydantic import BaseModel, Field
from typing import List


class RerankItem(BaseModel):
    """Individual reranked stock item"""
    stockCode: str = Field(..., description="종목 코드")
    stockName: str = Field(..., description="종목명")
    score: int = Field(..., ge=0, le=100, description="키워드 관련성 점수(0~100)")
    evidence: str = Field(..., description="context에서 찾은 근거 키워드/구절 1~2개 (짧게)")


class RerankResult(BaseModel):
    """List of reranked items"""
    items: List[RerankItem]


class NewsItem(BaseModel):
    """News item related to a stock"""
    title: str = Field(..., description="뉴스 제목")
    content: str = Field(..., description="뉴스 내용")
    link: str = Field(..., description="뉴스 링크")
    published_date: str = Field(..., description="발행 날짜")


class StockRecommendation(BaseModel):
    """Final stock recommendation - API response format"""
    name: str = Field(..., description="종목명")
    code: str = Field(..., description="종목 코드")
    description: str = Field(..., description="유사한 이유 설명 (AI가 작성함)")
    similarity: float = Field(..., ge=0.0, le=1.0, description="유사도 점수 (0.0~1.0)")
    news: List[NewsItem] = Field(default_factory=list, description="관련 뉴스 리스트")
