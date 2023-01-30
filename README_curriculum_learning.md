This branch has a few hacky modifications in order to enable generating datasets on different "levels", which can be used for curriculum learning. The main idea is to organize the grammar in different "levels" and generate different datasets for different levels (e.g., a lowest level for expressions, a second level for actions, a third level for individual sentence clauses, a fourth level for complex sentences constructed from multiple clauses).

# Changes
The changes are 
- `ocaml-grammars/ppx_deriving_random` contains a fork of `https://github.com/disteph/ppx_deriving_random.git#4.11`, which allows for additional level annotations: each constructor can be annotated with `[@level int]` and if a `set_level` property is given, only constructors with a matching level are generated.
- `ocaml-grammars/ppx_arsenal/src/ppx_deriving_arsenal.ml` has been modified to extend the random deriver with a `set_level` property.

To illustrate the use of the level annotations, consider the following example:
```ocaml
  type typ =
    | A of a  [@weight 5]
    | B of b  [@weight 1]
    [@@deriving arsenal]
```
This corresponds to the previous way of using `ppx_deriving_random`, i.e., constructor `A` is sampled with a weight 5 times as high as constructor `B`.

Now, as a first step, we can add optional level annotations:
```ocaml
  type typ =
    | A of a  [@weight 5] [@level 1]
    | B of b  [@weight 1] [@level 2]
    [@@deriving arsenal]
```

Adding these level annoations on its own won't change anything; since no `set_level` property is given, the random generator still behaves exactly as before. If we now add a `set_level` property to the arsenal deriver, i.e.,

```ocaml
  type typ =
    | A of a  [@weight 5] [@level 1]
    | B of b  [@weight 1] [@level 2]
    [@@deriving arsenal {set_level = 1}]
```

only constructors `A` will be generated. Accordingly, if we use `{set_level = 2}`, only constructors `B` will be generated. 

# Shortcomings

A major shortcoming of the current implementation is that only `int`s can be passed as the values for `set_level` (and not variables representing `int` values). Thus, generating datasets on different levels currently requires modifying the source code. Better approaches would be to (i) either support passing variables in the annotations (or extend the `state` concept with another property for `set_level`, or (ii) set a global variable in the random deriver.)