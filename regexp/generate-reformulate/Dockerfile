# This Dockerfile must be run from the top of arsenal2.0 so that 
# all the source directories are available
# Docker build doesn't allow for ../foo type of constructs
# at top of repo run 
#docker build . -t ${DOCKERNAME} -f regexp/generate-reformulate/Dockerfile
# or use the build.sh script
FROM ocaml/opam2:alpine AS reformulate-builder

# configure Opam version
RUN opam --version
RUN opam switch 4.07
RUN eval $(opam env)
RUN opam update

# install dependencies
RUN sudo apk add --no-cache m4 linux-headers ncurses && \
opam pin add -y ppx_deriving_random git+https://github.com/disteph/ppx_deriving_random.git
RUN mkdir -p /home/opam/reformulate/ppx_arsenal
RUN mkdir -p /home/opam/reformulate/arsenal_lib
#First the core needs
COPY --chown=opam:nogroup ocaml-grammars/ppx_arsenal/* /home/opam/reformulate/ppx_arsenal/
#must copy separately so they don't flatten 
COPY --chown=opam:nogroup ocaml-grammars/arsenal_lib/_tags /home/opam/reformulate/arsenal_lib/
COPY --chown=opam:nogroup ocaml-grammars/arsenal_lib/arsenal_lib.opam /home/opam/reformulate/arsenal_lib/
COPY --chown=opam:nogroup ocaml-grammars/arsenal_lib/src /home/opam/reformulate/arsenal_lib/src
#Now the regexp  - must copy separately so they don't flatten
COPY --chown=opam:nogroup regexp/generate-reformulate/arsenal_re.opam /home/opam/reformulate/
COPY --chown=opam:nogroup regexp/generate-reformulate/_tags /home/opam/reformulate/
RUN mkdir -p /home/opam/reformulate/src
COPY --chown=opam:nogroup regexp/generate-reformulate/src /home/opam/reformulate/src
RUN opam pin add -y ppx_deriving_arsenal /home/opam/reformulate/ppx_arsenal
#this also installs
RUN cd /home/opam/reformulate/arsenal_lib && \
opam pin -y arsenal_lib .
#cd only for the duration of the RUN line 
RUN cd /home/opam/reformulate && \ 
opam install ./arsenal_re.opam --deps-only
RUN eval $(opam env) && \
cd /home/opam/reformulate && \
ocamlbuild -use-ocamlfind src/REreformulate.native

# Multi-stage build, now we build the runtime image
FROM alpine:3.9
EXPOSE 8080
RUN apk add --no-cache tzdata && \
ln -s /usr/share/zoneinfo/UTC /etc/localtime && \
mkdir -p /opt/bin
COPY --from=reformulate-builder /home/opam/reformulate/REreformulate.native /opt/bin/REreformulate.native
CMD /opt/bin/REreformulate.native
