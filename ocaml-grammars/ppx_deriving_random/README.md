# ``ppx_deriving_random``

This syntax extension can add a random generator for a custom type
definition.  Adding ``[@@deriving random]`` to the type definition ``foo``
will provide a function of signature ``val random_foo : PPX_Random.state -> foo``. An additional argument
is prepended to the argument list for each type variable in the style of
other ``deriving`` extensions. That is,

```ocaml
type ('a1, ..., 'aN) foo = ... [@@deriving random]
```

will provide

```ocaml
val random_foo : (PPX_Random.state -> 'a1) ... (PPX_Random.state -> 'aN) ->
                 PPX_Random.state -> ('a1, ..., 'aN) foo
```

If the type being defined is called `t`,
then the name of the function is simply `random`,
consistently with other ``deriving`` extensions.

## Installation and Usage

If you use ``opam`` you can install ``ppx_deriving_random`` with

```shell
opam pin add ppx_deriving_random git+https://github.com/disteph/ppx_deriving_random.git
```
or with the usual variants of the URL to get different branches or tags, such as

```shell
git+https://github.com/disteph/ppx_deriving_random.git#4.11
```


For manual installation from a fresh Git checkout, make sure you have the dependencies indicated in `ppx_deriving_random.opam`, then

```shell
dune build
dune install
```
You can use it like any other ppx extension; if you want to use it in a project using dune, you can check the `dune` file in directory `examples`.

## Context

The ppx extension assumes that a module ``PPX_Random`` is in scope, with signature:

```ocaml
module PPX_Random : sig

  type state
  (* State passed to the random generators. *)

  val case : int -> state -> int
  (* A function used to select a random constructor or row field. The first
     argument is the number of cases. The result must be in the range from 0
     to the number of cases minus one and should be uniformly distributed. *)

  val case_30b : state -> int
  (* When fields are weighted (see below), this is used instead of
     random_case. In the returned value, bit 29 and downwards to the desired
     precision must be random and bit 30 upwards must be zero. *)

  val deepen : state -> state
  (* A function used to update the state when going one level down in the
     generation of the random AST. For instance if type state has a field depth,
     one can instantiate deepen to increase that field by one. *)

end
```

These can be bound to ``Random.State.t``, ``fun i s -> Random.State.int s i``,
``Random.State.bits``, and the identity function, respectively, but this is not
done by default since advanced applications may need to pass additional information
in ``PPX_Random.state`` or use different random number generators.


## Example and customization

```ocaml
type 'a free_magma =
  | Fm_gen of 'a
  | Fm_mul of 'a free_magma * 'a free_magma
  [@@deriving random]
```

will generate

```ocaml
val random_free_magma : (PPX_Random.state -> 'a) -> PPX_Random.state -> 'a free_magma
```

### Weights

However, there is a problem with the above generated function. By default all
branches of the variant are equally probable, leading to a possibly diverging
computation.  This can be fixed by weighing the constructors:

```ocaml
type 'a free_magma =
  | Fm_gen of 'a [@weight 3]
  | Fm_mul of 'a free_magma * 'a free_magma [@weight 2.5]
  [@@deriving random]
```

The probability that a constructor is picked when generating a random value of a
variant type is its weight divided by the sum of the weights over all
constructors.

A weight can be a static weight as above, i.e. an int or float constant which
defaults to 1 if not provided. It can also be a dynamic weight, i.e. an
expression of type ``PPX_Random.state -> float``, representing a weight that
depends on the current state. For instance if ``PPX_Random.state`` contains a
field ``depth`` recording how deep in the AST the generator is working (and
updated by function ``deepen`` as described above), then the probability that a
constructor is picked may depend on the depth. By recording the right
information in the state (e.g., by ``[@random expr]``), the probability that a
constructor be picked may depend on how many times it has already been picked on
the path from the root of the AST, or been picked in the constructed parts of
the AST so far, or been picked since forever. Biasing the probability
distribution with dynamic weights is also another way to avoid diverging
generation.

Note: if all weights of a variant type are static, then the probabilities are
computed statically during the PPX code generation, otherwise they are computed
at runtime at every call of the generator function.

### References to other types and using custom generators

If the type definition refers to other types, `ppx_deriving_random` will produce calls to correspondingly named generators. For instance,

```ocaml
type a = C of u 
[@@deriving random]
```
will produce a call to ``random_u`` (and again, a reference to type ``t`` will produce a call to ``random``).
Such a function should, at the place of the type definition,
be present in the scope,
either manually defined or itself generated by `ppx_deriving_random`.

If needed, the generator used for an occurrence of type ``u`` in the type definition can be overridden by a custom generator function.
Instead of the occurrence of `u` in the type definition, 
use `(u[@random expr])`, where ``expr`` is an expression of type ``PPX_Random.state -> u``.

If ``u`` is an instance of a polymorphic type, say ``foo bar``,
then `(foo bar[@random expr])` expects ``expr``
to be of type ``PPX_Random.state -> foo bar``.

Alternatively, you can use `(foo bar[@custom expr])`, which expects
``expr`` to be of type ``(PPX_Random.state -> foo) -> PPX_Random.state -> foo bar``;
``random_foo`` will be passed to it as argument.

These customization mechanisms allow the use of different generators (e.g., following different distributions) for different occurrences of the same type. Of course, another way to tweak the generation (e.g., the distribution) is to define type aliases with manually written custom random generators.
For instance, one can define

```ocaml
type u_alias = u
let random_u_alias = expr
```
before a type definition that now refers to `u_alias` instead of `u`.