FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY .git .git

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "vector_database.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--ssl-certfile=/app/certificates/fullchain.pem", \
     "--ssl-keyfile=/app/certificates/privkey.pem", \
     "--reload"]
