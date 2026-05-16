FROM python:3.10-slim

# Install system dependencies for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Spacy model for Presidio NLP
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
