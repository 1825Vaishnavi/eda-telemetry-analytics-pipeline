FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
    numpy==1.26.2 \
    pydantic==2.5.3 \
    scikit-learn==1.3.2

# Copy source and models
COPY src/ ./src/
COPY models/ ./models/

ENV MODEL_PATH=models/lstm_best.pt
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
