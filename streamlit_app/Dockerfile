FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Les fichiers de données seront montés via un volume
RUN mkdir -p /app/data

EXPOSE 8501

ENV API_URL=http://api:8000

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
