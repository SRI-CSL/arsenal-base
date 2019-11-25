This is a project for AST generation in OCaml.

It provides three (kinds of) tools,
each of which comes in the form of an executable file:

- The AST generators themselves,
printing generated AST on standard out in different formats,
namely Natural language, Polish notation, S-expressions, and JSON

- A translator from the Polish notation or S-expressions formats to the JSON format
(reads standard in, writes on standard out). Works for any of the generators.

- A reformulator webservice:
it listens on a port for HTTP POST requests that each provide a JSON AST,
it replies a number of natural language reformulations of the AST.
Specific to the regular expression domain (but can be easily adapted for others).


Here are the instructions to build and use:

=============
Preliminaries (for all tools)

System dependencies:

You will need
- the opam package manager for ocaml, version 2.0 which you may get from your distribution or from https://opam.ocaml.org/
- and an ocaml compiler >= 4.06 and < 4.08, which you can install via opam (see the `opam switch` command)

Arsenal core ocaml/opam dependencies:
You need to run
    opam pin add ppx_deriving_random git+https://github.com/disteph/ppx_deriving_random.git
and from arsenal2.0/ocaml-grammars
    opam pin add -y ppx_deriving_arsenal ./ppx_deriving_arsenal

and from arsenal2.0/ocaml-grammars/arsenal_lib
    opam pin -y arsenal_lib .
#which also completes the 'opam install arsenal_lib'

Optional:
To edit ocaml source files, it is convenient to use the `merlin` and `tuareg` tools, available via
   opam install tuareg merlin

=============
Building an AST generator (example below is for regular expressions):
You need to run

From arsenal2.0/regexp/generate-reformulate
    ocamlbuild -use-ocamlfind src/REgenerate.native

It will create the executable file `REgenerate.native`

Usage:
Examples are

    ./REgenerate.native
    ./REgenerate.native 15
    ./REgenerate.native -json
    ./REgenerate.native -sexp
    ./REgenerate.native 1000000 -json -polish -sexp -arity_sep "#"

to generate 10^6 examples nl, json, polish, sexp

Run `REgenerate.native -h` for all options
Use the -one-entity flag to only generate one entity type.

=============
Building the reformulation webservice (example below is for regular expressions):
After completing the preliminaries, you need to run

From arsenal2.0/regexp/generate-reformulate
    ocamlbuild -use-ocamlfind src/REreformulate.native

It will create the executable file `REreformulate.native`

Usage:
Examples are

    ./REreformulate.native
    ./REreformulate.native 8056

These will start the service.
Example of HTTP request:
    curl -H "howmany:20" -d "MY_JSON_STRING" -X POST "http://localhost:8056/"

howmany indicates how many natural language reformulations the service must produce.
It's the only header that the service looks at.
