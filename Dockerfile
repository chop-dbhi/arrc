FROM ubuntu:14.04

RUN apt-get update

RUN apt-get install -y python python-dev python-pip

ADD . /arrc

RUN apt-get install -y python-scipy

RUN pip install -r /arrc/requirements.txt

WORKDIR /arrc

EXPOSE 5000

CMD ["python", "rs.py"]

