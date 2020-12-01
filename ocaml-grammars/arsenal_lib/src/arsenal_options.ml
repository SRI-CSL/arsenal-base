open Sexplib
open Arg
open Arsenal_lib

module JSON = Yojson.Safe;;

(************************)
(* Command line options *)

let check_opt  = ref false (* Check outputs *)
let print_nl   = ref true  (* Output Natural Language by default *)
let print_json = ref false (* Do not output json by default *)
let print_sexp = ref false (* Do not output S-expression by default *)
let print_polish = ref false (* Do not output polish notation by default *)
let howmany = ref 1        (* Default number of example to generate is 1 *)
let args : string list ref = ref [] (* Where the command-line argument will be stacked *)

let options =
  [
    ("-json", Set print_json, "\tprint json output");
    ("-sexp", Set print_sexp, "\tprint S-expression output");
    ("-polish", Set print_polish, "\tprint polish output");
    ("-arity_sep", String(fun s -> Polish.arity := s), "\tin Polish notation, specify separator between token and its arity (default is '#')");
    ("-one-entity", Set Entity.one_entity_kind, "\tin natural language, only use 1 kind of entities");
    ("-types", Set PPX_Sexp.print_types, "\tdisplay types in generated data");
    ("-no-nl", Clear print_nl, "\tprint disable natural language output");
    ("-check", Set check_opt, "\tcheck output");
    ("-strict-entities", Float(fun f -> Entity.strict := f), "\thow strict entity kinds should be considered (0. : they are ignored; +infty: very strictly enforced)");
  ]

module Grammar = struct

  type 'a t = {
    random    : PPX_Random.state -> 'a;
    pp        : 'a pp;
    to_yojson : 'a -> JSON.t;
    sexp_of   : 'a -> Sexp.t;
  }

  let generate grammar n =
    let print_tab s = print_string ("\t"^s) in
    for i = 1 to n do
      let t = PPX_Random.init() |> grammar.random in
      (* if !check_opt  then check sentence_to_yojson sentence_of_yojson sexp_of_sentence sentence_of_sexp t; *)
      if !print_nl   then print_string(toString(grammar.pp t));
      if !print_json then print_tab(JSON.to_string(grammar.to_yojson t));
      if !print_sexp then print_tab(Sexp.to_string(grammar.sexp_of t));
      if !print_polish then print_tab(Polish.to_string(Polish.of_sexp(grammar.sexp_of t)));
      print_endline ""
    done

end
