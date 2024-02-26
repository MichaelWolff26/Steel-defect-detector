FROM python:3.11-slim-bookworm

WORKDIR /app

RUN pip install poetry

COPY . .

RUN poetry install

EXPOSE 8080

CMD poetry run streamlit run --server.port 8080 Streamlit_GUI.py --server.address 0.0.0.0


