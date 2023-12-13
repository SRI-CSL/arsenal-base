FROM continuumio/miniconda3:22.11.1
RUN pip install --upgrade pip
RUN pip install flask==3.0.0

RUN mkdir -p /opt/src
COPY process.py /opt/src
WORKDIR /opt/src
ENV PYTHONPATH=${PYTHONPATH}:/opt/src
EXPOSE 8080
CMD python -u process.py
