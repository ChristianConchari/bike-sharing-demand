# Base image for Apache Airflow
FROM apache/airflow:2.8.1

# Switch to root user to install additional packages
USER root

# Update the package list and install git
RUN apt-get update \
  && apt-get install -y --no-install-recommends git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy the requirements.txt file to the container
COPY requirements.txt /requirements.txt

# Upgrade pip and install Apache Airflow with specific extras and additional Python packages
RUN pip install --user --upgrade pip \
    && pip install --no-cache-dir --user apache-airflow[virtualenv]==2.8.1 -r /requirements.txt

# Copy data to the container
COPY train.csv /opt/airflow/data/train.csv