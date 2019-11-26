# Arsenal Regular Expression Demo

- [Overview](#overview)
- [Run from Docker Images](#run-from-docker-images)
- [Running the Demo](#running-the-demo)
- [Creating Local Docker Images](#creating-local-docker-images)
- [Running Components Manually](#running-components-manually)
- [Command Line Processing](#command-line-processing)
- [Output Example](#output-example)

## Overview
This directory contains an example of Arsenal applied to the natural language description
of regular expressions.

You can either run the demo using pre-canned Docker images or you can compile the source code
(as needed) and run on your native system.  

Whether running from Docker images or from native-built source code, you can process input text and
choose the output options with the command-line tool "gen_regexps.py".
When running from Docker images you can also choose to use a graphical user interface to take in
input and choose between various output options.

The custom components of this demo are:
- a regular expression domain entity processor to identify entities from natural language text, found in `regexp-entity/entity.py`
- a regular expression domain grammar and its pretty-printing functions as well as a current model
that has been trained on the current grammar and its pretty-printing functions, found in `generate-reformulate/src`
- an option GUI found in `regexp-ui/src`

The standard Arsenal components of this demo are:
- the NL to CST (natural language to syntax tree) engine that uses the defined model
- the underlying Arsenal base grammar support found in `../ocaml-grammars/arsenal_lib/src`

When running a model, Arsenal takes inputs that are natural-language (NL) sentences representing regular expression syntax
and gives outputs that are regular expressions as well as optionl concrete syntax trees (CSTs) or reformulations of the original input.

## Building the Docker Images

To build all the docker images for the regexp example, execute the command

  docker-compose build

in the current directory.

To rebuild a specific image, or several images but not all, append the corresponding
service names from the docker-compose file at the end of the command, e.g.,

  docker-compose build ui entity

## Running The Demo

To run the demo, from the regexp directory, execute the command

  docker-compose up

To use the UI, point a browser at http://localhost:8080.
Sample input is in the file "example.txt"

To use the command line interface gen_regexps.py, see the instructions for running
native local executables (INSERT LINK)

To quit a demo, hit ctrl+c in the same terminal (possibly several times),
and then

  docker-compose down -v

## Running Components Manually

It is possible to run the regexp example directly from locally built source code (without use of Docker containers)

NL2CST
The code is the Arsenal core and is found under <arsenal_root>/seq2seq and is written in Python and doesn't require additional compilation.

ENTITY PROCESSOR
The code is specific to the regular expression example and is found under <arsenal_root>/regexp/regexp-entity. It is written in Python and
doesn't require any additional compilation.

REFORMULATION
The code is specific to the regular expression example and is found under <arsenal_root>/regexp/generate-reformulate.
It is written in ocaml and relies on Arsenal core libraries found in <arsenal_root>/ocaml-grammars.

The defined grammar is used to generate a model that is used by the NL2CST processor. Most changes to the grammar would
require generating a new model to have Arsenal work properly.
If you make changes to either the core ocaml libraries or the regular expression grammar, you need to do the following.

# Rebuild the grammar component like this:
From the <arsenal_root>/regepx directory
eval $(opam env)
opam update
opam pin add -y ppx_deriving_random git+https://github.com/disteph/ppx_deriving_random.git
opam pin add -y ppx_deriving_arsenal ../ocaml-grammars/ppx_arsenal/
opam pin -y arsenal_lib ../ocaml-grammars/arsenal_lib/
cd generate-reformulate/
opam install ./arsenal_re.opam --deps-only
rm -rf _build

### How to start the components manually
Start up the components manually on the host system from the <arsenal_root>/regexp directory

```bash
$export MODEL_ROOT=./models/regexp_2019-09-26
$python ../seq2seq/src/run_nl2cst_server.py 8070 &
$python regexp-entity/server.py 8060 &
$generate-reformulate/REreformulate.native 8090 &
```

## Command Line Processing

The `gen_regexps.py` tool allows you to process input text at the command line.
The default output will be the regexp equivalent to the natural language input text.

```bash
$python gen_regexps.py example.txt
```

Optional output, which can be compbined:
```bash
$python gen_regexps.py example.txt -r  
```
> Show the reformulation with substitutions in addition to the regular expressions.

```bash
$python gen_regexps.py example.txt -c
```
> Show the CSTs generated from the natural language in addition to the regular expressions.

```bash
$python gen_regexps.py example.txt -e
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
