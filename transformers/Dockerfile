FROM continuumio/miniconda3:22.11.1
RUN pip install --upgrade pip
RUN pip install torch==2.1.1 datasets==2.15.0 transformers==4.35.2 flask==3.0.0 tqdm==4.64.1

RUN mkdir -p /opt/models && \
  mkdir -p /opt/src
# We split up the following to make the build/deploy faster when we only change the source code
COPY runtime /opt/src
WORKDIR /opt/src
ENV PYTHONPATH=${PYTHONPATH}:/opt/src
EXPOSE 8080
CMD python -u run_nl2cst_transformers.py
