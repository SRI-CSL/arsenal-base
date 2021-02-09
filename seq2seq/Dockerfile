FROM continuumio/miniconda3:4.7.12
RUN conda install -c pytorch pytorch-cpu && \
  conda clean --all
RUN conda install -c conda-forge \
  flask matplotlib tqdm tensorflow && \
  conda clean --all
RUN mkdir -p /opt/models && \
  mkdir -p /opt/src
# We split up the following to make the build/deploy faster when we only change the source code
COPY src /opt/src
WORKDIR /opt/src
ENV PYTHONPATH=${PYTHONPATH}:/opt/src
EXPOSE 8080
CMD python -u run_nl2cst_server.py
