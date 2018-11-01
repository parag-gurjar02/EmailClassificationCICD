
FROM ubuntu:16.04
MAINTAINER Mathew Salvaris <mathew.salvaris@microsoft.com>

#RUN mkdir /code
#WORKDIR /code
#ADD . /code/
#ADD etc /etc

RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        python \
        python-dev \
        python-setuptools \
        python-pip \
        supervisor \
        nginx && \
    #rm /etc/nginx/sites-enabled/default && \
    #cp /code/nginx/app /etc/nginx/sites-available/ && \
    #ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/ && \
    pip install -r /requirements.txt && \
    

EXPOSE 88
#CMD ["supervisord", "-c", "/etc/supervisord.conf"]
