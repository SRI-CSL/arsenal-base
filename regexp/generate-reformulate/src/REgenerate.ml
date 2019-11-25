open Arg
open Format
open Sexplib
open Std

open Arsenal_lib
open Arsenal_options
open REgrammar
open REgrammar_pp

let grammar = Grammar.{
      random    = random_re;
      pp        = pp_re;
      to_yojson = re_to_yojson;
      sexp_of   = sexp_of_re;
              }

let options =
  options @
  [
    ("-concat-max", Int(fun i -> concat_max := i), "n\tmaximal length of concatenations is n (default is "^string_of_int !concat_max);
    (* ("-string-max", Int(fun i -> string_max := i), "n\tmaximal length of random strings is n (default is "^string_of_int !string_max); *)
  ]

let description =
"This is an ocaml generator of abstract syntax trees for Regexps. The available options are below. The command takes, as optional argument, the number of examples to generate (default is 1).";;

Arg.parse options (fun a->args := a::!args) description;;

match !args with
| [n] -> howmany := int_of_string n
| [] -> ()
| _ -> failwith "Too many arguments in the command";;

Grammar.generate grammar !howmany
