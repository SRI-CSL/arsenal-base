FROM sricsl/arsenal-builder:0.0.18 as builder
COPY --chown=opam:nogroup generate-reformulate/ /home/opam/reformulate/reformulate/
RUN cd /home/opam/reformulate/reformulate && \
    eval $(opam env) && \
    dune build
 
FROM debian:11
EXPOSE 8080
RUN mkdir -p /opt/bin
COPY --from=builder /home/opam/reformulate/reformulate/_build/default/src/reformulate.exe /opt/bin/reformulate.exe
CMD /opt/bin/reformulate.exe 8080 -hide-entities -separator="/" -type-arg "_" ""  -json-type-arg  "_" "" -top="Grammar_effigy/Top/t" -json-path-mode 1 -json-separator "_"
