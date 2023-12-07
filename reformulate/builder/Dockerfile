
FROM ocaml/opam:debian-11-ocaml-4.11

### Install dependencies

# External dependencies
#RUN sudo apk add --no-cache m4 linux-headers ncurses
RUN sudo apt-get update --allow-releaseinfo-change
RUN sudo apt-get update
RUN sudo apt-get install m4 libgmp-dev -y

# configure Opam version
RUN opam --version
RUN opam repo add ocaml.org https://opam.ocaml.org --all-switches
RUN opam update
RUN opam switch list-available
RUN opam switch 4.11
RUN eval $(opam env)

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

#cd only for the duration of the RUN line 
RUN cd /home/opam/reformulate/ppx_arsenal && opam pin add -y ppx_deriving_arsenal .
RUN cd /home/opam/reformulate/arsenal_lib && opam pin add -y arsenal .
