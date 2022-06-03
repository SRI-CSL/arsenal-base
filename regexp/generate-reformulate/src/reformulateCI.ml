open Containers
open Sexplib
open Arsenal.Postprocess
open Arsenal.Arsenal_lib
open Arsenal.Arsenal_options
open Grammar.REgrammar
open Grammar.REgrammar_pp

let () = Grammar.REgrammar.load
let () = Grammar.REgrammar_pp.load

let json_separator = ref !separator
let json_str_arg   = ref !str_arg
let json_qualify_mode = ref !qualify_mode

let swap_ref a b = let c = !a in a := !b; b :=c

let rec regexp fmt =
  let open Format in
  function
  | StartOfLine re -> fprintf fmt "^%a" regexp re
  | EndOfLine re   -> fprintf fmt "%a$" regexp re
  | Plus(Terminal _ as re) -> fprintf fmt "%a+" regexp re
  | Plus re        -> fprintf fmt "(%a)+" regexp re
  | Star(Terminal _ as re) -> fprintf fmt "%a*" regexp re
  | Star re        -> fprintf fmt "(%a)*" regexp re
  | Or(re1, re2)   -> fprintf fmt "(%a)|(%a)" regexp re1 regexp re2
  | Concat l -> fprintf fmt "%a" (List.pp ~pp_sep:(fun _ () -> ()) regexp) l
  | Terminal terminal ->
     match terminal with
     | Specific s -> fprintf fmt "%s" (Entity.get_subst s)
     | Empty      -> fprintf fmt "\"\""
     | CharacterRange(a,z) -> fprintf fmt "[%s-%s]" (Entity.get_subst a) (Entity.get_subst z)
     | Word       -> fprintf fmt "\\w"
     | Any        -> fprintf fmt "."
     | Digit      -> fprintf fmt "\\d"
     | Space      -> fprintf fmt "\\s"
     | NotWord    -> fprintf fmt "\\W"
     | NotDigit   -> fprintf fmt "\\D"
     | NotSpace   -> fprintf fmt "\\S"

let cst_process serialise pp ?global_options ?options ?original ?ep ?cleaned ~id ~to_sexp csts =

  let cst = match csts with
    | `List((`String _)::_) -> csts
    | `List(((`List _) as cst)::_) -> cst
    | json -> raise (Conversion("Not a good json for csts; this should be a JSON array: "^(JSON.to_string json)))
  in
  let cst = cst |> to_sexp |> serialise.PPX_Serialise.of_sexp in 

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
  let dico =
    ("regexp",
     `String (Format.sprintf "%a" regexp cst))
    ::dico
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
  swap_ref json_separator separator;
  swap_ref json_str_arg str_arg;
  swap_ref json_qualify_mode qualify_mode;
  let dico =
    ("cst", serialise.PPX_Serialise.to_json cst) :: dico
  in
  swap_ref json_separator separator;
  swap_ref json_str_arg str_arg;
  swap_ref json_qualify_mode qualify_mode;
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

let options = Arsenal.Arsenal_options.options

let args = ref []

let () = Arg.parse options (fun arg -> args := arg :: !args) (description "Command-line reformulator")

let run serialise pp filename =
  postprocess (cst_process serialise pp) (JSON.from_file filename)
    
let run filename = run Grammar.REgrammar.serialise_re pp_re filename

let () = match !args with
  | [filename] -> run filename |> JSON.pretty_print Format.stdout
  | _ -> failwith "Expecting exactly one argument"
