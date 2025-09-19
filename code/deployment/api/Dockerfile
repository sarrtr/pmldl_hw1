FROM python:3.12-slim

WORKDIR /app

RUN pip install fastapi uvicorn numpy scikit-learn==1.3.2 joblib

COPY ./code/deployment/api /app
COPY ./models /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]