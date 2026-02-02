"""
테스트 스크립트: Vector DB 구축 및 검색 테스트
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.vector_store_service import VectorStoreService
from config import Config

def test_vector_db():
    print("=" * 70)
    print("Vector DB 구축 테스트")
    print("=" * 70)
    
    try:
        # Config 검증
        Config.validate()
        print("✅ Config 검증 완료\n")
        
        # Vector Store 초기화
        print("Vector Store 초기화 시작...")
        vector_store = VectorStoreService()
        print()
        
        # 테스트 검색
        test_keywords = ["반도체", "배터리", "화장품"]
        
        for keyword in test_keywords:
            print(f"\n🔍 키워드: '{keyword}'")
            print("-" * 70)
            
            results = vector_store.similarity_search_with_score(keyword, k=5)
            
            print(f"검색 결과 (Top 5):")
            for idx, (doc, distance) in enumerate(results, 1):
                m = doc.metadata
                print(f"  {idx}. [{m['name']}] ({m['code']}) - {m['industry']}")
                print(f"     Distance: {distance:.4f}")
            print()
        
        print("=" * 70)
        print("✅ 모든 테스트 완료!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
        return False
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_vector_db()
    sys.exit(0 if success else 1)
