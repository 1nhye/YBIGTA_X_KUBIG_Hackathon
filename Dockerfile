# 베이스 이미지 설정 (Python 3.10 버전 사용)
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY modules/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY ./modules /app/modules

# 엔트리포인트 설정 (필요시 main.py를 직접 실행할 수 있음)
CMD ["python", "/app/modules/main.py"]
