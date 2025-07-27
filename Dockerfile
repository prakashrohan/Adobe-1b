FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpoppler-cpp-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# ⬇️ Install spaCy model the correct way
RUN python -m spacy download en_core_web_md

# Copy your project files
COPY . .

# Run your pipeline (you can override in docker run)
CMD ["python", "main.py", "input/input.json", "output/output.json"]
