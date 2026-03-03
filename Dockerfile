# Gold Price Prediction - Hugging Face Spaces (Docker)
FROM python:3.10-slim

WORKDIR /app

# System dependencies (libgomp for LightGBM / XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy outputs and models (required for app and for generating plots)
COPY outputs/ outputs/
COPY models/ models/

# Copy only static assets that are not binary (style.css). PNGs are generated in next step.
COPY static/ static/

# Generate PNG plots at build time (avoids pushing binary files to Hugging Face)
COPY generate_analysis_plots.py .
RUN python generate_analysis_plots.py

# Copy application
COPY app.py .
COPY templates/ templates/

# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Run with gunicorn for production
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
