open Sexplib

open Arsenal.Postprocess
open Arsenal.Arsenal_lib
open Arsenal.Arsenal_options
open Grammar.REgrammar_pp

let () = Grammar.REgrammar.load
let () = Grammar.REgrammar_pp.load

let () = deterministic := true

let cst_process serialise pp ?global_options ?options ?original ~id ~to_sexp csts =
  (* Polish JSON list -> list of Polish JSONs *)
  let csts = match csts with
    | `List csts -> csts
    | json -> raise (Conversion("Not a good json for csts; this should be a JSON array: "^(JSON.to_string json)))
  in
  (* Build from the list of Polish JSONs a list of 'a; ill-typed Polish JSONS are skipped *)
  let cst, i =
    let rec aux ?e i = function
      | [] ->
         begin
           match e with
           | Some e -> raise e
           | None -> raise(Conversion("Reformulate: Didn't get a single Polish sequence from NL2CST"))
         end
      | cst::tail ->
         try
           (cst |> to_sexp |> serialise.PPX_Serialise.of_sexp), i
         with
           e -> aux ~e (i+1) tail
    in
    aux 0 csts
  in

  (* now we construct the json dictionary for the output *)
  let dico = [] in
  (* constructing S-expression if asked *)
  let dico =
    if !print_sexp
    then
      ("s-exp",
       `String (CCFormat.sprintf "%a"
                  Sexp.(if !pretty then pp_hum else pp_mach)
                  (serialise.PPX_Serialise.to_sexp cst)))
      ::dico
    else dico
  in
  (* constructing reformulation if asked *)
  let dico =
    if !print_nl
    then
      ("reformulation",
       `String (CCFormat.sprintf "%a" (fun fmt t -> pp t fmt) cst))
      ::dico
    else dico
  in
  (* outputting original text if given *)
  let dico = match original with
    | None -> dico
    | Some o -> ("orig-text", `String o) :: dico
  in
  (* outputting the json and number of outputs skipped *)
  let dico =
    ("cst", serialise.PPX_Serialise.to_json cst) :: ("skipped_csts", `Int i) :: dico
  in
  let dico =
    let id = match id with
      | `String s -> s
      | json -> raise (Conversion("Not a good json for an id (should be a string: "^(JSON.to_string json)))
    in
    let id = match global_options with
      | None -> id
      | Some l ->
         match JSON.Util.member "prefix" (`Assoc l) with
         | `String prefix -> prefix ^ id
         | json -> raise (Conversion("Not a good json for a prefix (should be a string: "^(JSON.to_string json)))
    in
    ("sentence_id", `String id) :: dico
  in
  `Assoc dico

let top = ref "REgrammar/re"

let options =
  [
    ("-top", Arg.String(fun s -> top := s), "n\ttop-level type (default is \"REgrammar/re\")");
    ("-nondeterministic", Clear deterministic, "\tDeterministic NL generation (default is false)");
    ("-hide-entities", Clear Entity.ppkind, "\tif a substitution string is present in entity, just show the string");
    ("-json", Set print_json, "\tprint json output");
    ("-no-nl", Clear print_nl, "\tprint disable natural language output");
    ("-one-entity", Set Entity.one_entity_kind, "\tin natural language, only use 1 kind of entities");
    ("-pretty", Set pretty, "\tprint S-expressions in human-readable form");
    ("-sexp", Set print_sexp, "\tprint S-expression output");
    ("-short", Set short, "\tcontructors and entity kinds have short strings and are not prefixed by module names");
    ("-verb", Int(fun i -> verb := i), "\tverbosity level (default is 0)");

  ]

let () = Arg.parse options (fun arg -> port := int_of_string arg) (description "RE grammar")

let run serialise pp = Lwt_main.run (main ~port:!port (cst_process serialise pp))

let run a = match Register.find_opt a with
  | Some (About{ key; serialise; _ }) -> run serialise !(TUID.get_pp key)
  | None -> run Grammar.REgrammar.serialise_re pp_re

let () =
  run !top
