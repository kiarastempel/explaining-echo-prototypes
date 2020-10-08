FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests python python3-pip libgl1-mesa-glx -y
RUN pip3 install --upgrade pip
COPY requirements_docker.txt .
RUN pip3 install -r requirements.txt
ENTRYPOINT  ["python3", "src/train.py"]
