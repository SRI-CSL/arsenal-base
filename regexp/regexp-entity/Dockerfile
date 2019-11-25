FROM continuumio/miniconda3
RUN conda install -c conda-forge web.py cheroot && conda clean --all
WORKDIR /opt
ADD . /opt
ENV PYTHONPATH=${PYTHONPATH}:/opt
CMD python -u server.py
