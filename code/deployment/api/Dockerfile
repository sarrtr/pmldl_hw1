# FROM python:3.12-slim

# WORKDIR /app

# RUN pip install --no-cache-dir fastapi uvicorn scikit-learn numpy joblib

# COPY ./code/deployment/api/ /app/

# COPY ./models/ /app/models/

# ENV MODEL_DIR=/app/models

# EXPOSE 8000

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.12-slim

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn numpy joblib

COPY ./code/deployment/api/ /code/deployment/api/

COPY ./models/ /models/

EXPOSE 8000

CMD ["python3", "/code/deployment/api/app.py"]