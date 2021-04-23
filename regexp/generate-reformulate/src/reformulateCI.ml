open Containers
open Arsenal.Postprocess
open Arsenal.Arsenal_lib
open Grammar.REgrammar_pp

let () = Grammar.REgrammar.load
let () = Grammar.REgrammar_pp.load

let cst_process serialise pp ?global_options ?options ?original ~id cst =
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
  ("-top", Arg.String(fun s -> top := s), "n\ttop-level type (default is \"sentence\")")
  ::Arsenal.Arsenal_options.options

let args = ref []

let () = Arg.parse options (fun arg -> args := arg :: !args) (description "Command-line reformulator")

let run serialise pp filename =
  postprocess serialise (cst_process serialise pp) (JSON.from_file filename)
    
let run filename = match Register.find_opt !top with
  | Some (About{ key; serialise; _ }) ->
     debug 1 "Picking top-level type to be %s@," (TUID.name key);
     run serialise !(TUID.get_pp key) filename
  | None ->
     debug 1 "Picking top-level type to be %s (should be %s)@," (Grammar.REgrammar.serialise_re.typestring()) !top;
     run Grammar.REgrammar.serialise_re pp_re filename

let () = match !args with
  | [filename] -> run filename |> JSON.pretty_print Format.stdout
  | _ -> failwith "Expecting exactly one argument"
