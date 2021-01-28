open Sexplib
open Arsenal_lib

(************************)
(* Command line options *)

val check_opt : bool ref  (* Check outputs *)
val print_nl : bool ref   (* Output Natural Language by default *)
val print_json : bool ref (* Do not output json by default *)
val print_sexp : bool ref (* Do not output S-expression by default *)
val print_polish : bool ref (* Do not output polish notation by default *)
val howmany : int ref (* Default number of example to generate is 1 *)
val args : string list ref (* Where the command-line argument will be stacked *)
val options : (string * Arg.spec * string) list

module Generate_runtime : sig

  type 'a t = {
    random    : PPX_Random.state -> 'a;
    pp        : 'a pp;
    to_yojson : 'a -> JSON.t;
    sexp_of   : 'a -> Sexp.t;
  }

  val generate : 'a t -> int -> unit
    
end
