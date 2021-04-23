open Containers
open Arsenal
open Arsenal_lib
open Arsenal_options

let () = Grammar.REgrammar.load
let () = Grammar.REgrammar_pp.load

let description =
"This is an ocaml generator of abstract syntax trees for RE. The available options are below. The command takes, as optional argument, the number of examples to generate (default is 1).";;

let () = Arg.parse options (fun a->args := a::!args) description;;

let treat a = match Register.find_opt a with
  | Some about -> generate about !howmany
  | None -> 
     Format.(fprintf stderr) "@[<v>Could not find a type called @[<v>%s@]@]@," a;
     failwith "FAIL"

let () =  
  match !args |> List.rev with
  | [a] -> treat a
  | [] ->
     Format.(fprintf stderr) "@[<v>Top type not specified, you must choose among@,  @[<v>%a@]@]@,"
       (List.pp String.pp) (Register.all());
     failwith "FAIL"
  | _::_ ->
     Format.(fprintf stderr) "@[<v>You can only specify one top type@]@,";
     failwith "FAIL"
