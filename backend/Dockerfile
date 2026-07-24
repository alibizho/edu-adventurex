# Backend API server. Built by docker-compose.yml at the repo root. The ml-service is NOT in this
# image — it runs on Hyper AI (GPU); point the backend at it via ML_SERVICE_URL in backend/.env.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
