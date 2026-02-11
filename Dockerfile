# Usando imagem base Python
FROM python:3.11-slim

# Atualiza o apt e instala FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia arquivos do projeto
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expõe porta que você usa no main.py
EXPOSE 8000

# Comando para rodar o app
CMD ["python", "main.py"]


# Start do servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

