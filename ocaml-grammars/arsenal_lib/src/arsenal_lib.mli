open Containers
open Format

open Sexplib

(* Useful abbreviation *)
module JSON = Yojson.Safe

(************************************************************)
(* Conversions between json, S-expressions, Polish notation *)

exception Conversion of string

module PPX_Serialise : sig
  type 'a t = {
      to_json : 'a -> JSON.t;
      to_sexp : 'a -> Sexp.t;
      of_sexp : Sexp.t -> 'a;
      type_string : string;
    }
  val print_null  : bool ref (* Does not print null values in JSON *)
  val json_cons   : (string * JSON.t) -> (string * JSON.t) list -> (string * JSON.t) list
  val print_types : bool ref (* Print types in S-expressions? *)
  val sexp_constructor : string -> string -> Sexp.t
  val sexp_throw    : ?who:tag -> Sexp.t -> _
  val sexp_is_atom  : Sexp.t -> bool
  val sexp_get_cst  : ?who:tag -> Sexp.t -> tag
  val sexp_get_type : ?who:tag -> Sexp.t -> tag
end

module JSONindex : sig
  type t
  val mem : string -> bool
  val find : string -> (string*JSON.t) list
  val mark : string -> t
  val add : t -> (string*JSON.t) list -> unit
  val out : id:string -> description:string -> JSON.t
end
     
val sexp2json : Sexp.t -> JSON.t
val exn : ('a -> ('b, string) result) -> 'a -> 'b

(* A module for Polish notation *)
module Polish : sig
  val arity : string ref
  val of_sexp : Sexplib.Sexp.t -> string list
  val to_sexp : (string * int) list -> Sexplib.Sexp.t
  val to_string : string list -> string
  val of_string : arity:Char.t -> string -> (string * int) list
end

(********************************************************)
(* Lists and options: conversions to/from S-expressions *)

val typestring_bool: string
val typestring_int : string
val typestring_list: string -> string
val typestring_option: string -> string

val json_desc_list: string -> unit -> unit
val json_desc_option: string -> unit -> unit

val serialise_bool   : bool PPX_Serialise.t
val serialise_int    : int PPX_Serialise.t
val serialise_list   : ('a PPX_Serialise.t) -> 'a list PPX_Serialise.t
val serialise_option : ('a PPX_Serialise.t) -> 'a option PPX_Serialise.t

(******************)
(* For random AST *)
    
module PPX_Random : sig
  type state
  type 'a t = state -> 'a
  val deepen : state -> state
  val case : int -> state -> int
  val case_30b : state -> int
  val init : unit -> state
end
val depth             : PPX_Random.state -> float
val random_int        : int PPX_Random.t
val random_bool       : bool PPX_Random.t
val random_ascii      : char PPX_Random.t
val random_ascii_pair : (char * char) PPX_Random.t
val random_option     : ?p:float -> ('b PPX_Random.t) -> 'b option PPX_Random.t
val ( +? ) : ('b PPX_Random.t) -> float -> 'b option PPX_Random.t
val ( *? ) : ('b PPX_Random.t) -> ('c PPX_Random.t) -> ('b * 'c) PPX_Random.t
val random_list :
  ?min:int -> ?max:int -> ?empty:float -> ('b PPX_Random.t) -> 'b list PPX_Random.t
val string_of_char : char -> string
val random_string : string PPX_Random.t
type 'a distribution = ('a*float) list
    
(****************)
(* For printing *)

type print = Format.formatter ->  unit
type 'a pp = 'a -> print

val (^^)   : print -> print -> print
val return : string pp
val noop   : print

type 'a formatted =
  | F : ('a , Format.formatter, unit) format -> 'a formatted
  | FormatApply : ('a -> 'b) formatted * 'a  -> 'b formatted
  | Noop : unit formatted

val print : 'a formatted -> formatter -> 'a
val (//)  : ('a -> 'b) formatted -> 'a  -> 'b formatted

val pick  : ('a * int) list -> 'a

val toString : print -> string

val pp_list : ?sep:string -> ?last:string -> 'a pp -> 'a list pp


(* Easy choose *)

(* For lists of strings, possibly weighted random pick *)
val ( !! )  : string list pp
val ( !!! ) : (string * int) list pp

(* For lists of %t functions *)
val ( !~ )  : print list pp
val ( !~~ ) : (print * int) list pp

(* For lists of anything *)
val ( !& )  : 'a list -> _ -> 'a
val ( !&& ) : ('a * int) list -> _ -> 'a

(* For lists of formatted *)
val ( !? )  : 'a formatted list -> Format.formatter -> 'a
val ( !?? ) : ('a formatted * int) list -> Format.formatter -> 'a

(* Easy weights *)
val ( ?~ ) : bool -> int (* true -> 1 | false -> 0 *)
val ( ~? ) : bool -> int (* true -> 0 | false -> 1 *)
val ( ??~ ) : 'a option -> int (* has option -> 1 else 0 *)
val ( ~?? ) : 'a option -> int (* has option -> 0 else 1 *)
val ( ++ ) : int -> int -> int (* Logical OR *)

(* Easy extension of pp function to option type *)
val ( +? ) : 'a pp -> print -> 'a option pp
val ( ?+ ) : 'a pp -> 'a option pp (* noop if not present *)

(* has the option? *)
val ( ?++ ) : 'a option -> bool


(*************************************************)
(* Small extension of the standard Result module *)

module Result : sig
  include module type of Result
  val of_option : error:'a -> 'b option -> ('b, 'a) result
  val bind : ('a, 'b) result -> ('a -> ('c, 'b) result) -> ('c, 'b) result
  val map  : ('a, 'b) result -> ('a -> 'c) -> ('c, 'b) result
end

val ( >>= ) : ('a, 'b) Result.result -> ('a -> ('c, 'b) Result.result) -> ('c, 'b) Result.result
val ( >>| ) : ('a, 'b) Result.result -> ('a -> 'c) -> ('c, 'b) Result.result

(***********************)
(* Module for entities *)

module Entity :
sig
  val one_entity_kind : bool ref
  val warnings : [ `NoSubst of string ] list ref
  val strict   : float ref

  type 'a t [@@deriving arsenal]
  val pp    : (Format.formatter -> 'a -> unit) -> 'a t pp
  val to_id : string -> (string * int, string) result
  val entity_mk :
    string ->
    ?kind:'a option ->
    ?counter:int ->
    ?substitution:string option -> unit -> ('a t, 'b) Result.result
  val pick : ('a * float) list -> _ -> 'a
end

