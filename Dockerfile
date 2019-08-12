FROM python:3.7.2-stretch

ADD /data/ /opt/data/
ADD /bin/ /opt/bin/

RUN apt-get update -y

RUN pip3 install scipy
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install gensim
RUN pip3 install pattern
RUN pip3 install nltk

CMD ['sleep', 'infinity']
