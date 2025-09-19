# FROM python:3.12-slim

# WORKDIR /app

# RUN pip install --no-cache-dir streamlit requests

# COPY ./code/deployment/app/ /app/

# ENV STREAMLIT_SERVER_PORT=8501
# ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# EXPOSE 8501

# CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]


FROM python:3.12-slim

RUN pip install --no-cache-dir streamlit requests

COPY ./code/deployment/app/ /code/deployment/app/

CMD ["streamlit", "run", "/code/deployment/app/streamlit_app.py"]