
# To build this, go to the parent directory and run
#   docker-compose build reformulate
FROM ocaml/opam2:4.11 AS reformulate-builder

# configure Opam version
RUN opam --version
RUN opam repo add ocaml.org https://opam.ocaml.org --all-switches
RUN opam update
RUN opam switch list-available
RUN opam switch 4.11
RUN eval $(opam env)

### Install dependencies

# External dependencies
#RUN sudo apk add --no-cache m4 linux-headers ncurses
RUN sudo apt-get update
RUN sudo apt-get install m4 libgmp-dev -y
RUN opam depext conf-pkg-config
RUN opam pin add -y ppx_deriving_random git+https://github.com/disteph/ppx_deriving_random.git#4.11

# Local dependencies
# Creating directories in the image
RUN mkdir -p /home/opam/reformulate/ppx_arsenal
RUN mkdir -p /home/opam/reformulate/arsenal_lib
RUN mkdir -p /home/opam/reformulate/reformulate

#Copying opam dependency list
COPY --chown=opam:nogroup ocaml-grammars/arsenal_dependencies.opam /home/opam/reformulate/
RUN cd /home/opam/reformulate && opam install ./arsenal_dependencies.opam --deps-only

#Copying material
COPY --chown=opam:nogroup ocaml-grammars/ppx_arsenal/ /home/opam/reformulate/ppx_arsenal/
COPY --chown=opam:nogroup ocaml-grammars/arsenal_lib/ /home/opam/reformulate/arsenal_lib/
#Now the 5G specific material
COPY --chown=opam:nogroup regexp/generate-reformulate/ /home/opam/reformulate/reformulate/

#cd only for the duration of the RUN line 
RUN cd /home/opam/reformulate/ppx_arsenal && opam pin add -y ppx_deriving_arsenal .
RUN cd /home/opam/reformulate/arsenal_lib && opam pin add -y arsenal .
RUN cd /home/opam/reformulate/reformulate && opam install ./arsenal_re.opam --deps-only
RUN cd /home/opam/reformulate/reformulate && \
    eval $(opam env) && \
    dune build

# Multi-stage build, now we build the runtime image
# FROM alpine:3.9
# EXPOSE 8080
# RUN apk add --no-cache tzdata && \
# ln -s /usr/share/zoneinfo/UTC /etc/localtime && \
# mkdir -p /opt/bin
# COPY --from=reformulate-builder /home/opam/reformulate/reformulate/_build/default/src/reformulate.exe /opt/bin/reformulate.exe
# CMD /opt/bin/reformulate.exe

# Alpine no longer works with recent OCaml versions.
FROM debian:stable
EXPOSE 8080
RUN mkdir -p /opt/bin
COPY --from=reformulate-builder /home/opam/reformulate/reformulate/_build/default/src/reformulate.exe /opt/bin/reformulate.exe
CMD /opt/bin/reformulate.exe 8080 -top "$TOP"