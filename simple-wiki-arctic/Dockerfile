FROM embed-base:latest

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
WORKDIR /code/app

VOLUME ["/root/data"]
VOLUME ["/root/models"]

CMD ["uvicorn", "simple-wiki-arctic:app", "--host", "0.0.0.0", "--port", "80"]

