# ``ppx_deriving_random``

This syntax extension can add a random generator for a custom type
definition.  Adding ``[@@deriving random]`` to the type definition ``foo``
will emit ``val random_foo : PPX_Random.state -> foo``.  An additional argument
is prepended to the argument list for each type variable in the style of
other ``deriving`` extensions.  That is,

```ocaml
type ('a1, ..., 'aN) foo = ... [@@deriving random]
```

will provide

```ocaml
val random_foo : (PPX_Random.state -> 'a1) ... (PPX_Random.state -> 'aN) ->
                 PPX_Random.state -> ('a1, ..., 'aN) foo
```

## Installation and Usage

If you use ``opam`` you can install ``ppx_deriving_random`` with

```ocaml
opam pin add ppx_deriving_random git+https://github.com/disteph/ppx_deriving_random.git
```
or with the usual variants of the URL to get different branches or tags.


For manual installation from a fresh Git checkout, make sure you have OASIS
and ``ppx_deriving``, then
```shell
oasis setup -setup-update dynamic
ocaml setup.ml -configure --prefix your-prefix
make
make install
```
where ``your-prefix/lib`` is in your findlib path.  The syntax extension can
now be enabled with ``-package ppx_deriving_random.ppx`` or by putting
``package(ppx_deriving_random.ppx)`` in your ``_tags``.  To build the
example:
```shell
ocamlfind ocamlopt -linkpkg -package ppx_deriving_random.ppx -package ppx_deriving.show free_magma.ml
```

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

References to other types will produce additional calls to correspondingly
named generators. You should provide generators with suitable distributions
for your usage in scope of the type definition. If needed, the generator
used for a particular type reference ``u`` can be overridden with
``[@random expr]``, where ``expr : PPX_Random.state -> u``. This allows using
different distributions for different instances of the same type. Another
way to tweak the distribution is to define type aliases with custom
random generators.

## Example

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

However, there is a problem with the generated function.  By default all
branches of the variant is equally probable, leading to a possibly diverging
computation.  This can be fixed by weighing the constructors:

```ocaml
type 'a free_magma =
  | Fm_gen of 'a [@weight 3]
  | Fm_mul of 'a free_magma * 'a free_magma [@weight 1.5]
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
