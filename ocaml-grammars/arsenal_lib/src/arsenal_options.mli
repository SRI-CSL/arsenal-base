open Arsenal_lib

(************************)
(* Command line options *)

val check_opt : bool ref    (* Check outputs *)
val print_nl : bool ref     (* Output Natural Language by default *)
val print_json : bool ref   (* Do not output json by default *)
val print_sexp : bool ref   (* Do not output S-expression by default *)
val print_polish : bool ref (* Do not output polish notation by default *)
val pretty   : bool ref     (* prettyfy S-expressions *)
val howmany  : int ref      (* Default number of example to generate is 1 *)
val raw_json : bool ref     (* Whether json should just be like Sexp *)
val args     : string list ref  (* Where the command-line argument will be stacked *)
val options  : (string * Arg.spec * string) list
  
val generate : About.t -> int -> unit
