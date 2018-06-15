FROM python:3.5

RUN apt-get update
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl && \
    pip3 install torchvision

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#ADD ./src /opt/src
#ADD ./model_weights /opt/model_weights

WORKDIR /opt/src