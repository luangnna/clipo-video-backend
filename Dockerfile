FROM python:3.11-slim

# Dependências de sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp
RUN pip install --no-cache-dir yt-dlp

WORKDIR /app

# Dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Código
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "main.py"]
