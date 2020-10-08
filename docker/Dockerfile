FROM tensorflow/tensorflow:2.0.0-gpu

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests python python3-pip -y
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY /src /src
WORKDIR /src
