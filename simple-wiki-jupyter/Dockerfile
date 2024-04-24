FROM embed-base:latest

RUN useradd -ms /bin/bash jupyter

USER jupyter
WORKDIR /home/jupyter

ENV PATH="${PATH}:/home/jupyter/.local/bin"

COPY requirements.txt /
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

EXPOSE 8888

VOLUME /home/jupyter/notebooks
VOLUME /home/jupyter/models
VOLUME /home/jupyter/data

ENTRYPOINT [ "jupyter" , \
    "notebook",\
    "--allow-root",\
    "--ip=0.0.0.0",\
    "--port=8888",\
    "--no-browser",\
    "--NotebookApp.token=''",\
    "--NotebookApp.password=''",\
    "--notebook-dir=/home/jupyter/notebooks" ]
