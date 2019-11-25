open Lwt

open Reformulate
open REgrammar
open REgrammar_pp;;
    
Arg.parse [] (fun arg -> port := int_of_string arg) (description "formal regular expressions");;
Lwt_main.run (main ~port:!port re_of_yojson pp_re)
