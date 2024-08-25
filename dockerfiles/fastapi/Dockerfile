# Use a lightweight version of Python 3.8 as the base image
FROM python:3.8-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host machine to the /app directory in the container
COPY . .

# Copy the requirements.txt file separately to leverage Docker's caching mechanism for dependencies
COPY requirements.txt /requirements.txt

# Update package list and install curl
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version to ensure compatibility with packages
RUN pip install --upgrade pip

# Install Python dependencies listed in requirements.txt without using the cache to reduce image size
RUN pip install --no-cache-dir -r /requirements.txt

# Expose port 8800 for the application
EXPOSE 8800