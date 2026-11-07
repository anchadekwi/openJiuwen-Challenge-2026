FROM python:3.11.10-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run_agent.py .
COPY harness ./harness
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
