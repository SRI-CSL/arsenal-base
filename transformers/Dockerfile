FROM continuumio/miniconda3:4.7.12
RUN pip install --upgrade pip
RUN pip install torch datasets flask tqdm

# begin custom transformers installation
# (adds typeforcing functionality and fixes a dependency specification)
ARG tr_version="4.4.2"
RUN wget https://github.com/huggingface/transformers/archive/v$tr_version.tar.gz
RUN tar -xf v$tr_version.tar.gz -C /opt
COPY typeforcing_patch/* /opt/transformers-$tr_version/src/transformers/
RUN pip install -e /opt/transformers-$tr_version
# end custom transformers installation

RUN mkdir -p /opt/models && \
  mkdir -p /opt/src
# We split up the following to make the build/deploy faster when we only change the source code
COPY runtime /opt/src
WORKDIR /opt/src
ENV PYTHONPATH=${PYTHONPATH}:/opt/src
EXPOSE 8080
CMD python -u run_nl2cst_transformers.py
