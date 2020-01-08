FROM continuumio/miniconda3:4.7.12
RUN conda update openssl ca-certificates certifi
RUN conda install -c conda-forge web.py cheroot && conda clean --all
WORKDIR /opt
ADD . /opt
ENV PYTHONPATH=${PYTHONPATH}:/opt
CMD python -u server.py
