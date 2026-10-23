FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# train a model and replay the demo once so the API has a champion to serve, then start the API
CMD ["sh", "-c", "python run_demo.py && uvicorn mlmon.api:app --host 0.0.0.0 --port 8000"]
