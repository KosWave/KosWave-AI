from flask import Flask, jsonify
from routes.stock_routes import stock_bp
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

app.register_blueprint(stock_bp, url_prefix='/api')

@app.route('/health')
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({'status': 'healthy', 'service': 'KosWave Stock API'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
