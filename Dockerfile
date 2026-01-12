FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        pkg-config \
        perl \
        libterm-readline-perl-perl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "vector_database.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--ssl-certfile=/app/certificates/fullchain.pem", \
     "--ssl-keyfile=/app/certificates/privkey.pem", \
     "--reload"]
