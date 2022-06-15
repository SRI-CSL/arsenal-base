# Arsenal Regular Expression Demo

- [Overview](#overview)
- [Transformer-related changes](#prerequisiteschanges-to-get-the-transformers-based-nl2cst-component-running)
- [Run from Docker Images](#run-from-docker-images)
- [Running the Demo](#running-the-demo)
- [Creating Local Docker Images](#creating-local-docker-images)
- [Running Components Manually](#running-components-manually)
- [Command Line Processing](#command-line-processing)
- [Output Example](#output-example)

## Overview
This directory contains an example of Arsenal applied to the natural language description
of regular expressions.

At runtime, Arsenal takes inputs that are natural-language (NL) sentences describing regular expressions and produces outputs that are regular expressions as syntax trees as well as their reformulations in natural language.

You can either run the demo using Docker images or you can compile the source code (as needed) and run on your native system.  

Whether running from Docker images or from native-built source code, you can process input text and
choose the output options with the command-line tool `gen_regexps.py`.
When running from Docker images you can also choose to use a graphical user interface to take in
input and choose between various output options.

The domain-specific components of this demo are:

- An entity processor that identifies regular expression entities (strings and characters) in natural language, found in `regexp-entity/entity.py`;
- A machine learning model trained to translate a natural language description of a regular expression into a syntax tree for it (the tree is produced in Polish notation), found in `models/2021-06-24T1637-07_41c4757_REgrammar-re_2021-06-24T1816-07`;
- A reformulation service that turns the Polish notation of a syntax tree into an actual tree in JSON format, accompanied by a new natural language rendering of it (for verification purposes), found in `generate-reformulate`;
- An optional GUI found in `regexp-ui/src`.

The regular expression trees are trees that are well-formed according to a grammar that was specifically and manually designed for regular expressions, and that can also be found in `generate-reformulate`. That component also provides pretty-printing functions into natural language, and a generator of training datasets (pairs of tree + natural language renderings), used to train the model.

The domain generic Arsenal components of this demo are:

- the underlying Arsenal base grammar and pretty-printer, found in `../ocaml-grammars/arsenal_lib`;
- an NL to CST (natural language to syntax tree) engine that reads and runs any trained model, found in `../seq2seq`.

## Prerequisites/changes to get the transformers-based nl2cst component running
- Get the trained model archive file (from...?) and extract into `./models/transformers/`. This should create a new subfolder `06-07-2022` with the trained model. `docker-compose.yml` is set up to use that model.
- Building the dockerized reformulator (cf. `generate-reformulate/docker/README.md` for more details):
  1. in `generate-reformulate/docker/builder`: run `./build.sh`
  2. in `generate-reformulate/docker/reformulator`: run `./build.sh`
- After this, the other docker images can be build and started as described below.

## Building the Docker Images

To build all the docker images for the regexp example, execute the command

```bash
docker compose build
```
in the current directory.

To rebuild a specific image, or several images but not all, append the corresponding
service names from the docker-compose file at the end of the command, e.g.,

```
docker compose build ui entity
```

## Running The Demo

To run the demo, from the regexp directory, execute the command

```
docker compose up
```
To use the UI, point a browser at http://localhost:8080.
Sample input is in the file `example.txt`.

To use the command line interface gen_regexps.py, see the instructions for [running native local executables](#command-line-processing).

To quit a demo, hit ctrl+c in the same terminal (possibly several times),
and then

```
docker compose down -v
```

## Running Components Manually

It is possible to run the regexp example directly from locally built source code (without use of Docker containers).

### NL2CST:
The code is the Arsenal core and is found under `<arsenal_root>/seq2seq` and is written in Python and doesn't require additional compilation.

### Entity Processor:
The code is specific to the regular expression example and is found under `<arsenal_root>/regexp/regexp-entity`. It is written in Python and
doesn't require any additional compilation.

### Generation and reformulation:
The code is specific to the regular expression example and is found under `<arsenal_root>/regexp/generate-reformulate`.
It is written in ocaml and relies on Arsenal core libraries found in `<arsenal_root>/ocaml-grammars`.

The defined grammar is used to generate a model that is used by the NL2CST processor. Most changes to the grammar would
require generating a new model to have Arsenal work properly.
If you make changes to either the core ocaml libraries or the regular expression grammar, you need to re-build the grammar component as follows.

From the `<arsenal_root>/regexp` directory:

```bash
eval $(opam env)
opam update
opam pin add -y ppx_deriving_random git+https://github.com/disteph/ppx_deriving_random.git#4.11
opam pin add -y ppx_deriving_arsenal ../ocaml-grammars/ppx_arsenal/
opam pin add -y arsenal ../ocaml-grammars/arsenal_lib
cd generate-reformulate/
opam install ./arsenal_re.opam --deps-only
dune build
```

### How to start the components manually
Start up the components manually on the host system from the <arsenal_root>/regexp directory

```bash
export MODEL_ROOT=./models/2021-06-24T1637-07_41c4757_REgrammar-re_2021-06-24T1816-07/
python ../seq2seq/src/run_nl2cst_server.py 8070 &
python regexp-entity/server.py 8060 &
generate-reformulate/_build/default/src/reformulate.exe 8090 &
```

## Command Line Processing

The `gen_regexps.py` tool allows you to process input text at the command line.
The default output will be the regexp equivalent to the natural language input text.

```bash
python gen_regexps.py example.txt
```

Optional output, which can be combined:
```bash
python gen_regexps.py example.txt -r  
```
> Show the reformulation with substitutions in addition to the regular expressions.

```bash
python gen_regexps.py example.txt -c
```
> Show the CSTs generated from the natural language in addition to the regular expressions.

```bash
python gen_regexps.py example.txt -e
```
> Show the interim JSON with the entity subsitutions in addition to the regular expressions.


## Output Example

Running the gen_regexps.py process using an innput file with just one sentence in it and using all the output flags
generates first the interim (debuggin) "ENTITIES" version of the input sentence with the original text and any regular-expression matching
entities pulled out and replaced with substitions.
The concrete syntax tree version of the sentence is shown is in the CSTs" section of the output.  This output shows the entity-replaced version
of the input text and then the CST that represents that sentence.  For more information about the format of CSTs, see the
[Understanding Concrete Syntax Trees document](../doc/Understanding-Concrete-Syntax-Trees.md)
The regular expression interpretation of the input text is followed by a randomized generated language version of the regular expression as defined by the REreformulate.native executable.

```bash
$cat one.txt
the string "abc", followed by any space character
$python gen_regexps.py -r -c -e

ENTITIES:
{
  "sentences": [
    {
      "id": "S1",
      "orig-text": "the string \"abc\", followed by any space character",
      "new-text": "the string String_1, followed by any space character",
      "substitutions": {
        "String_1": "\"abc\""
      }
    }
  ]
}
CSTs:
{
  "sentences": [
    {
      "id": "S1",
      "nl": "the string String_1, followed by any space character",
      "cst": {
        "node": "Concat",
        "type": "re",
        "subtrees": [
          [
            {
              "node": "Terminal",
              "type": "re",
              "subtrees": [
                {
                  "node": "Specific",
                  "type": "terminal",
                  "subtrees": [
                    {
                      "node": {
                        "placeholder": "String_1",
                        "text": "\"abc\""
                      },
                      "type": "entity<kstring>",
                      "subtrees": []
                    }
                  ]
                }
              ]
            },
            {
              "node": "Terminal",
              "type": "re",
              "subtrees": [
                {
                  "node": "Space",
                  "type": "terminal",
                  "subtrees": []
                }
              ]
            }
          ]
        ]
      }
    }
  ]
}
Regular Expressions:
(abc\s)
string String_1{"abc"}, and then a space character
```
