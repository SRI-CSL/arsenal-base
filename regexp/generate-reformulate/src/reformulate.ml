open Arsenal.Postprocess
open Arsenal.Arsenal_lib
open Grammar.REgrammar_pp

let () = Grammar.REgrammar.load
let () = Grammar.REgrammar_pp.load

let () = deterministic := true

let cst_process serialise pp ?global_options ?options ?original ~id ~to_sexp cst =
  let cst = cst |> to_sexp |> serialise.PPX_Serialise.of_sexp in
  let reformulation =
    [
      "reformulation", `String (CCFormat.sprintf "%a" (fun fmt t -> pp t fmt) cst) 
    ]
  in
  let tail = match original with
    | None -> reformulation
    | Some o -> ("orig-text", `String o) :: reformulation
  in
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
  `Assoc (("sentence_id", `String id) :: ("cst", serialise.PPX_Serialise.to_json cst) :: tail)

let top = ref "REgrammar/re"

let options =
  [
    ("-top", Arg.String(fun s -> top := s), "n\ttop-level type (default is \"re\")");
  ]

let () = Arg.parse options (fun arg -> port := int_of_string arg) (description "RE grammar")

let run serialise pp = Lwt_main.run (main ~port:!port (cst_process serialise pp))

let run a = match Register.find_opt a with
  | Some (About{ key; serialise; _ }) -> run serialise !(TUID.get_pp key)
  | None -> run Grammar.REgrammar.serialise_re pp_re

let () = run !top
