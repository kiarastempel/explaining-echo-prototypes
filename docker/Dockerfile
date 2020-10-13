FROM tensorflow/tensorflow:2.0.0-gpu-py3

COPY requirements_docker.txt .
RUN pip3 install -r requirements_docker.txt
WORKDIR /src
ENTRYPOINT  []
CMD ["python", "train.py"]
