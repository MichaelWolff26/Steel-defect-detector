FROM python:3.11-slim-bookworm

WORKDIR /app

RUN pip install poetry

COPY . .

RUN poetry install

EXPOSE 8000


CMD ["poetry","run","uvicorn", "FastAPI:app", "--host", "0.0.0.0", "--port", "8000","--reload" ]

