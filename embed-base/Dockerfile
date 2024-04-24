FROM python:3.10

# Install packages & python base
RUN apt update && \
    apt upgrade -y && \
    apt install -y python3-pip

COPY requirements.txt /
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
