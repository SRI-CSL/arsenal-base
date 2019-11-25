# Expressing Conditions

In order to help with developing a domain-specific grammar, this tutorial gives examples of constructs for 3 kinds of conditions:

1. [Boolean Combination of Conditions](#boolean-combination-of-conditions)
2. [Temporal Aspects of Conditions](#temporal-aspects-of-conditions)
3. [Conditions for at least N items](#conditions-for-at-least-n-items)

These constructs extend the expressivity of rules to include conditions to handle more copmlex natural language syntax - e.g. "if `[CONDITION]`, then
`[CONCLUSION]` should be deduced".

In order to illustrate the first two kinds of conditions - [boolean combinations of conditions](#boolean-combination-of-conditions) and [temporal aspects of conditions](#temporal-aspects-of-conditions), we assume we start from a grammar that feature a simple notion of condition and that we extend to those more sophisticated forms.

The third kind of condition - [conditions for for at least N items](#conditions-for-at-least-n-items) - can be expressed with built-in features of Arsenal grammar.  An example of an expression using these features is given.  

## Boolean Combination of Conditions

As an example, define a grammar type condition as a syntax tree of type `condition`, which is either a noun phrase or a verb phrase:
```ocaml
type condition =
  | ConditionNoun of condition_noun*bool
  | ConditionVerb of condition_verb
[@@deriving arsenal]
```
A boolean combination allows conjunctions and disjunctions of condition nouns or verbs in a list.
A modified definition with a boolean combination of condition verbs is as follows:
```ocaml
type combo_kind = Conjunction | Disjunction
[@@deriving arsenal]

type condition =
  | ConditionNoun of condition_noun*bool
  | ConditionVerb of combo_kind * (condition_verbs list)
[@@deriving arsenal]
```
Here, the single-value type `condition_verb` is replaced by a list type `condition_verbs`. A new argument, of type `combo_kind`, indicates whether the conditions in the list are combined as a conjunction or a disjunction. 

Along with type definitions, indications can be given of how to generate lists of conditions. Without any indication added to `type combo_kind`, the random generator is allowed decide whether a generated list is conjunctive or disjunctive, with equal (50%)
probabilities. But an indication can be given about how long the lists will be in the generated data, as follows:
```ocaml
type condition =
  | ConditionNoun of condition_noun*bool
  | ConditionVerb of combo_kind * (condition_verbs list[@random random_list ~min:1 ~empty:0.8 random_condition_verbs])
[@@deriving arsenal]
```
On this indication, the generator always generates at least one condition; then with probability `0.8` there will be one condition and with probability `0.2` two conditions or more; then with probability `0.2^2` there will be three or more; then with probability  `0.2^4` there will be four or more; and so on.  Note that later probabilities of one or more conditions are relative to earlier probabilities, so that absolute later probabilities are multiplied by earlier probabilities.

More complex conditions with boolean combinations can be expressed in natural
language, by definitions in the supporting pretty-printing file. 

For the initial definition, that supports just a condition that was either a ConditionNoun or ConditionVerb, this could be:
```ocaml
let pp_condition pp_arg a : condition pp = fun fmt c ->
  match c with
  | ConditionNoun(cond,b) -> ...
  | ConditionVerb cond ->
    let pp_c = pp_condition_verb in
    !?[
      lazy (fprintf fmt "%t %a, %a" incase pp_c cond pp_arg a);
      lazy (fprintf fmt "%a %t %a" pp_arg a incase pp_c cond);
      lazy (fprintf fmt "if %a, %a" pp_c cond pp_arg a);
      lazy (fprintf fmt "if %a, then %a" pp_c cond pp_arg a);
      lazy (fprintf fmt "%a if %a" pp_arg a pp_c cond);
    ]
```
For the more complex grammar definition that supports conjunction and disjunction, the pretty printing definition would be  modified to:
```ocaml
let pp_condition pp_arg a : condition pp = fun fmt c ->
  match c with
  | ConditionNoun(cond,b) -> ...
  | ConditionVerb(kind,condlist) ->
    let sep = "," in
    let last = match kind with
      | Conjunction -> " and"
      | Disjunction -> " or"
    in
    let pp_c = pp_list ~sep ~last pp_condition_verb in
    !?[
      lazy (fprintf fmt "%t %a, %a" incase pp_c condlist pp_arg a);
      lazy (fprintf fmt "%a %t %a" pp_arg a incase pp_c condlist);
      lazy (fprintf fmt "if %a, %a" pp_c condlist pp_arg a);
      lazy (fprintf fmt "if %a, then %a" pp_c condlist pp_arg a);
      lazy (fprintf fmt "%a if %a" pp_arg a pp_c condlist);
    ]
```
Here, the function `pp_list` provided in [`arsenal_lib.ml`](../ocaml-grammars/arsenal_lib/src/arsenal_lib.ml`) is used.  The value of the argument `~sep` is the string to use for separating the items in the list, except
for the last two items in the list; the value of the argument `~last` is the string to use for separating the last two items. For the former, a comma separator `,` is used.  For the latter, the separator depends on whether the combination is a conjunction or disjunction, so that the definition matches the value of `kind` to the value of either `Conjunction` or `Disjunction`. If the match is to `Conjunction`, the separator is "and"; if the match is to `Disjunction`, the separator is "or".  The last argument of `pp_list` is the function used for pretty-printing each item of the list, which is `pp_condition_verb`, as before.

## Temporal Aspects of Conditions

For language where it is important to be able to specify temporal aspects of conditions, the definition of `condition` might be modified like this:
```ocaml
type time_cond =
  | Within of q_time
  | For of q_time
[@@deriving arsenal]

type condition =
  | ConditionNoun of condition_noun*bool
  | ConditionVerb of combo_kind
                     * (condition_verb list[@random random_list ~min:1 ~empty:0.8 random_condition_verb])
                     * time_cond option[@random random_time_cond +? 0.3]
[@@deriving arsenal]
```

Here, an optional argument of type `time_cond option` is added to `ConditionVerb`, to specify the temporal aspect of the condition.

The type `time_cond` is also defined here so as to represent temporal information for each
condition in the list: `Within` represents a time window within which the condition
need to be met; `For` represents the duration of the condition.
Both constructors expect an argument specifying the time period, of type
`q_time`, which is one of the types provided in the base Arsenal grammar defined in the 
[`base_grammar.ml`](../ocaml-grammars/arsenal_lib/src/base_grammar.ml) file, with a pretty-printing function provided.

For the argument of type `time_cond option` added to the constructor
`ConditionVerb`, an indication is added for the random generator, which
must decide whether or not to generate the optional argument.  Writing `[@random
random_time_cond +? 0.3]` indicates that the argument is provided with probability 0.7, , using function `random_time_cond`, and is *not* provided with probability 0.3.

Finally, pretty-printing is adapted to the new definitions.
First, a pretty-printing function for type `time_cond` is defined as follows:
```ocaml
let pp_time_cond : time_cond pp = fun fmt c ->
  match c with
  | Within time -> !??[
      lazy (fprintf fmt "within %a" pp_q_time time), 3;
      lazy (fprintf fmt "within a period of %a" pp_q_time time), 2;
      lazy (fprintf fmt "within a time window of %a" pp_q_time time), 1;
    ]
  | For time -> !?[
      lazy (fprintf fmt "for %a" pp_q_time time);
      lazy (fprintf fmt "for a period of %a" pp_q_time time);
    ]
```
This pretty-printing function calls the pretty-printing function `pp_q_time`.

The function `pp_condition` is then adapted:
```ocaml
let pp_condition pp_arg a : condition pp = fun fmt c ->
  match c with
  | ConditionNoun(cond,b) -> ...
  | ConditionVerb(kind,condlist,timecond) -> ...
    let sep = "," in
    let last = match kind with
      | Conjunction -> " and"
      | Disjunction -> " or"
    in
    let pp_c = pp_list ~sep ~last pp_condition_verb in
    match timecond with
    | None ->  [AS_BEFORE]
    | Some time -> !?[
        lazy (fprintf fmt "%t %a %a, %a" incase pp_c condlist pp_time_cond time pp_arg a);
        lazy (fprintf fmt "%a %t %a %a" pp_arg a incase pp_c condlist pp_time_cond time);
        lazy (fprintf fmt "%a %t, %a, %a" pp_arg a incase pp_time_cond time pp_c condlist);
        lazy (fprintf fmt "if %a %a, %a" pp_time_cond time pp_c condlist pp_arg a);
        lazy (fprintf fmt "if, %a, %a, %a" pp_time_cond time pp_c condlist pp_arg a);
        lazy (fprintf fmt "if %a %a, then %a" pp_time_cond time pp_c condlist pp_arg a);
        lazy (fprintf fmt "if, %a, %a, then %a" pp_time_cond time pp_c condlist pp_arg a);
        lazy (fprintf fmt "%a if %a %a" pp_arg a pp_time_cond time pp_c condlist);
        lazy (fprintf fmt "%a if, %a, %a" pp_arg a pp_time_cond time pp_c condlist);
      ]
```

For information on compiling a syntax tree generator method to create data for training a model, look at the 
[`regexp`](../regexp/README.md) example.

## Conditions for at least N items

There is support in the built-in features of Arsenal grammar for evaluating language for expressions that contain at least `N` of a certain type of condition.
The options can be seen in the [`base_grammar.ml`](../ocaml-grammars/arsenal_lib/src/base_grammar.ml) file.
For example, the built-in function `MoreThan` expresses the condition of at least `N` reported condition nouns from the list `l` of possible conditions.  

> insert grammar example here?
