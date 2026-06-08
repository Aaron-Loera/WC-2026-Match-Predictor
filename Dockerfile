FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies first
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Picks up models, predictions, odds history, etc
COPY . .

EXPOSE 8000

# Command to run when the container starts
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
