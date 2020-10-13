FROM tensorflow/tensorflow:latest-gpu

COPY requirements_docker.txt .
RUN pip3 install -r requirements_docker.txt
RUN  useradd tensorflow
USER tensorflow
WORKDIR /src
CMD ["python", "train.py"]
