# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/requirements.txt

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y wget \
    && pip install --upgrade pip \
    && pip install mlflow[extras] psycopg2-binary boto3 cryptography pymysql \
    && pip install --no-cache-dir -r /app/requirements.txt

# Expose port 5000
EXPOSE 5000

# Command to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0"]
