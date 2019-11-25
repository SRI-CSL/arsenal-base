FROM continuumio/miniconda3
RUN conda install spacy && conda clean --all
RUN conda install -c conda-forge flask && conda clean --all
RUN python -m spacy download en
WORKDIR /opt
ADD . /opt
ENV PYTHONPATH=${PYTHONPATH}:/opt
CMD python -u server.py
