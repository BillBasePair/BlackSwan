FROM python:3.13-slim

# Install system dependencies for pandas, matplotlib, and viennarna
RUN apt-get update && apt-get install -y \
    build-essential \
    libpng-dev \
    libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Run the app
CMD ["streamlit", "run", "BlackSwan.py", "--server.port", "8501", "--server.address", "0.0.0.0"]