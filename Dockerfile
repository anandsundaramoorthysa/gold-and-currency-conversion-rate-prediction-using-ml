# Gold Price Prediction - Hugging Face Spaces (Docker)
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and pre-built assets (run prepare_for_huggingface.py locally first)
COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY outputs/ outputs/
COPY models/ models/

# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Run with gunicorn for production
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
