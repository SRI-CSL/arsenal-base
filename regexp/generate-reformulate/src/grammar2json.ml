open Containers
open Arsenal.Arsenal_lib
open Arsenal.Arsenal_options

let () = Grammar.REgrammar.load
let () = Grammar.REgrammar_pp.load

let description = "This is an exporter of the RE grammar into a JSON schema; give the top types as arguments"

let sentence_info =
  [
    "sentence_id",   `Assoc [ "type",   `String "string";
                              "format", `String "uri-reference" ];
    "s-exp",         `Assoc [ "type", `String "string" ];
    "reformulation", `Assoc [ "type", `String "string" ];
    "orig-text",     `Assoc [ "type", `String "string" ];
    "regexp",        `Assoc [ "type", `String "string" ];
  ]

let () =
  Arg.parse options (fun a->args := a::!args) description;
  List.iter JSONindex.populate !args;
  let toptype = match !args |> List.rev with
    | toptype::_ -> toptype
    | [] -> 
     Format.(fprintf stderr) "@[<v>Top type not specified, you must choose among@,  @[<v>%a@]@]@,"
       (List.pp String.pp) (Register.all());
     failwith "FAIL"
  in
  JSONindex.out
    ~id:"arsenal4RE.json"
    ~description:"Schema representing Arsenal grammar for RE."
    ~toptype
    ~sentence_info
  |> Yojson.Safe.pretty_to_string
  |> print_endline
