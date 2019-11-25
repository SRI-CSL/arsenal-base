open Sexplib
open Std
    
open Generate
open REgrammar

let description = "converts sexp to json for well-formed regular expressions";;
let filename = ref "";;
Arg.parse [] (fun a-> filename := a) description;;
let l = Macro.load_sexps !filename;;

let aux sexp =
  (* sexp |> re_of_sexp |> re_to_yojson |> Yojson.Safe.to_string |> print_endline;; *)
  sexp |> json_of_sexp |> Yojson.Safe.to_string |> print_endline;;
List.iter aux l;;
