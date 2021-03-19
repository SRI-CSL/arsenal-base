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
let raw_json = ref false   (* Whether json should just be like Sexp *)
let args : string list ref = ref [] (* Where the command-line argument will be stacked *)

let options =
  [
    ("-n", Int(fun i -> howmany := i), "\thow many instances to generate (default is 1)");
    ("-json", Set print_json, "\tprint json output");
    ("-sexp", Set print_sexp, "\tprint S-expression output");
    ("-polish", Set print_polish, "\tprint polish output");
    ("-raw-json", Set raw_json, "\tmaximal Produce raw JSONs (i.e. JSON versions of S-expressions) (default is false)");
    ("-arity_sep", String(fun s -> Polish.arity := s), "\tin Polish notation, specify separator between token and its arity (default is '#')");
    ("-one-entity", Set Entity.one_entity_kind, "\tin natural language, only use 1 kind of entities");
    ("-types", Set PPX_Serialise.print_types, "\tdisplay types in generated data");
    ("-no-nl", Clear print_nl, "\tprint disable natural language output");
    ("-check", Set check_opt, "\tcheck output");
    ("-strict-entities", Float(fun f -> Entity.strict := f), "\thow strict entity kinds should be considered (0. : they are ignored; +infty: very strictly enforced)");
  ]

module Generate_runtime = struct

  type 'a t = {
    random    : PPX_Random.state -> 'a;
    pp        : 'a pp;
    to_json   : 'a -> JSON.t;
    to_sexp   : 'a -> Sexp.t;
  }

  let generate grammar n =
    let print_tab s = print_string ("\t"^s) in
    for _ = 1 to n do
      let t = PPX_Random.init() |> grammar.random in
      (* if !check_opt  then check sentence_to_yojson sentence_of_yojson sexp_of_sentence sentence_of_sexp t; *)
      if !print_nl   then t |> grammar.pp |> toString |> print_string;
      if !print_json then t |> grammar.to_json |> JSON.to_string |> print_tab;
      if !raw_json   then t |> grammar.to_sexp |> sexp2json |> JSON.to_string |> print_tab;
      if !print_sexp then t |> grammar.to_sexp |> Sexp.to_string |> print_tab;
      if !print_polish then t |> grammar.to_sexp |> Polish.of_sexp |> Polish.to_string |> print_tab;
      print_endline ""
    done

end
