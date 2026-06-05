FROM python:3.12-slim

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

CMD ["python", "scripts/railway_trade.py"]
