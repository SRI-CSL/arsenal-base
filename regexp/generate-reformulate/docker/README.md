# Effigy-grammar docker files

This directory contains three subdirectories, containing files
necessary to build the following module:

- `builder` A shared builder image, that compiles the OCaml binaries
- `generator` Training data generator
- `service` Grammar service

To build and deploy the builder image, use the scripts in the `builder` directory.
This needs to be rebuilt whenever the dependencies have changed.

To build and run the service, use `docker-compose build` and
`docker-compose run` in the parent directory.

To build, deploy, and run the generator, use the scripts in the `generator` directory.

