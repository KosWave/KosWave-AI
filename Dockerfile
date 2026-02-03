# Ubuntu 20.04 LTS 기반 Python 3.13 이미지
FROM python:3.13-slim-bookworm

# 메타데이터
LABEL maintainer="KosWave-AI"
LABEL description="Stock Recommendation API Server"

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 작업 디렉토리 생성
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# data 디렉토리 생성 (ChromaDB 저장용)
RUN mkdir -p /app/data/chroma_db

# 포트 노출
EXPOSE 5000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health', timeout=5)" || exit 1

# gunicorn으로 프로덕션 실행
CMD ["gunicorn", "--bind", "0.0.0.0:5000",  "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
