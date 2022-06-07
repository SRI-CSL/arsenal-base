FROM sricsl/regex-builder:0.0.1 as builder
COPY --chown=opam:nogroup generate-reformulate/ /home/opam/reformulate/reformulate/
RUN cd /home/opam/reformulate/reformulate && \
    eval $(opam env) && \
    dune build

FROM debian:stable
EXPOSE 8080
RUN mkdir -p /opt/bin
COPY --from=builder /home/opam/reformulate/reformulate/_build/default/src/reformulate.exe /opt/bin/reformulate.exe
CMD /opt/bin/reformulate.exe 8080 -hide-entities 
