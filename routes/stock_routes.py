from flask import Blueprint, request, jsonify
from services.stock_recommendation_service import StockRecommendationService
import traceback

stock_bp = Blueprint('stock', __name__)
stock_service = StockRecommendationService()

@stock_bp.route('/company', methods=['GET'])
def search_stocks():
    """
    키워드로 유사한 주식 검색
    
    Query Parameters:
        keyword (str): 검색 키워드
        
    Returns:
        JSON 배열: [{"name": str, "code": str, "description": str, "similarity": float}]
    """
    try:
        keyword = request.args.get('keyword')
        
        if not keyword:
            return jsonify({'error': 'keyword is required'}), 400
        
        # 추천 결과 가져오기 (배열 형태)
        recommendations = stock_service.get_recommendations(keyword)
        
        # 배열 그대로 반환
        return jsonify(recommendations), 200
        
    except FileNotFoundError as e:
        return jsonify({
            'error': 'Stock data file not found',
            'detail': str(e)
        }), 500
        
    except Exception as e:
        print(f"❌ Error in search_stocks: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'detail': str(e)
        }), 500