# Set the Python version
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Update the package list and install wget
RUN apt-get update && apt-get install -y --no-install-recommends wget

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install MLflow with extras and other required packages
RUN pip install mlflow[extras] psycopg2-binary boto3 cryptography pymysql

# Copy the rest of the application code
COPY . /app

# Expose port 5000, the default port for MLflow
EXPOSE 5000

# Command to run the MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0"]