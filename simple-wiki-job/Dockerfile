FROM embed-base:latest

COPY requirements.txt /
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY app/main.py /root/

VOLUME /root/models
VOLUME /root/files

ENTRYPOINT ["python", "/root/main.py"]
