open Containers
open Format

open Sexplib

(**********************************************)
(* Managing debug, global vars and exceptions *)
(**********************************************)

type 'a printf = ('a, Format.formatter, unit) format -> 'a
val verb  : int ref      (* Verbosity level *)
val debug : int -> 'a printf
val exc   : ?stdout:bool
            -> (string -> exn)
            -> ?margin:int
            -> ('a, Format.formatter, unit, 'b) format4 -> 'a

(* Separator of fully qualified names (types and constructors); default is "." *)
val separator : string ref
val str_arg   : (string*string) ref

(* Controls syntax of fully qualified names for types and constructors *)
(* Some i: prunes the first i levels of the prefix; use i = 0 for the fully qualified name *)
(* None  : only retains the last part of the fully qualified name, i.e. no prefix *)
(* Default is Some 0 *)
val qualify_mode : int option ref

(* Hashtables for strings, used several times *)
module Stbl : CCHashtbl.S with type key = string

(*******************************************************************)
(* Mandatory environment for ppx_random, which produces random AST *)
(*******************************************************************)

module PPX_Random : sig
  type state
  type 'a t = state -> 'a
  val deepen : state -> state
  val case : int -> state -> int
  val case_30b : state -> int
  val init : unit -> state
end

val depth : PPX_Random.state -> float

(************************************************************)
(* Conversions between json, S-expressions, Polish notation *)
(************************************************************)

(* Useful abbreviation *)
module JSON = Yojson.Safe

exception Conversion of string
val raise_conv : ('a, Format.formatter, unit, 'b) format4 -> 'a

module PPX_Serialise : sig
  type 'a t = {
      to_json : 'a -> JSON.t;
      to_sexp : 'a -> Sexp.t;
      of_sexp : Sexp.t -> 'a;
      hash    : 'a Hash.t;
      compare : 'a Ord.t;
      typestring : unit -> string;
    }
  val constructor_qualify : (?mode: int option -> path: string list -> string -> string) ref
  val type_qualify        : (?mode: int option -> path: string list -> string -> string) ref
  val arg_name            : (is_list:bool -> arguments:int -> int -> string) ref
  val str_apply   : string -> string -> string
  val print_null  : bool ref (* Does not print null values in JSON *)
  val json_cons   : (string * JSON.t) -> (string * JSON.t) list -> (string * JSON.t) list
  val print_types : bool ref (* Print types in S-expressions? *)
  val json_constructor_field : string (* Name of JSON field that contains a grammar constructor *)
  val sexp_constructor : string -> string -> Sexp.t
  val sexp_throw    : who:tag -> Sexp.t -> _
  val sexp_is_atom  : Sexp.t -> bool
  val sexp_get_cst  : who:tag -> Sexp.t -> tag
  val sexp_get_type : who:tag -> Sexp.t -> tag
end

val sexp2json : Sexp.t -> JSON.t
val exn : ('a -> ('b, string) result) -> 'a -> 'b

(* A module for Polish notation *)
module Polish : sig
  val arity : string ref
  val of_sexp : Sexplib.Sexp.t -> string list
  val to_sexp : (string * int) list -> Sexplib.Sexp.t
  val to_string : string list -> string
  val of_list   : string list -> (string * int) list
  val of_string : arity:Char.t -> string -> (string * int) list
end

(****************************)
(* Printing functionalities *)
(****************************)

type print = Format.formatter -> unit
type 'a pp = 'a -> print

val deterministic : bool ref

val (^^)   : print -> print -> print
val return : string pp
val noop   : print

type 'a formatted =
  | F : ('a , Format.formatter, unit) format -> 'a formatted
  | FormatApply : ('a -> 'b) formatted * 'a  -> 'b formatted
  | Noop : unit formatted

val print : 'a formatted -> formatter -> 'a
val (//)  : ('a -> 'b) formatted -> 'a  -> 'b formatted

(* val pick  : ('a * int) list -> 'a *)

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
val ( ?? ) : 'a option -> bool


(**********************************************************************)
(* Type Unique ID                                                     *)
(* Registering type t produces a TUID for t, a value of type t TUID.t *)
(**********************************************************************)

(* Comparison result type for polymorphic types *)
module Order : sig
  type (_,_) t =
    | Eq : ('a,'a) t
    | Lt : ('a,'b) t
    | Gt : ('a,'b) t
end

module TUID : sig
  type _ t
  val create (* Creating TUID for type 'a requires providing hash and compare functions for 'a *)
      : hash: 'a Hash.t -> compare : 'a Ord.t -> random : 'a PPX_Random.t -> name:string -> 'a t
  val hash        : 'a t Hash.t                      (* Hashing a TUID *)
  val compare     : 'a t -> 'b t -> ('a, 'b) Order.t (* Comparing TUIDs *)
  val name        : 'a t -> string    (* name of 'a *)
  val get_hash    : 'a t -> 'a Hash.t (* Given a TUID for 'a, get a hash function for 'a *)
  val get_compare : 'a t -> 'a Ord.t  (* Given a TUID for 'a, get a compare function for 'a *)
  val get_pp      : 'a t -> 'a pp ref (* Given a TUID for 'a, get a pretty-printer for 'a *)
  val get_random  : 'a t -> 'a PPX_Random.t ref (* Given a TUID for 'a, get a random generator for 'a *)
end

(*************************************)
(* The registration system for types *)
(*************************************)

(* We keep a global mapping from strings (identifying types)
   to a record values containing information about the type.
   The type's pretty-printer can be modified, hence the ref *)

module About : sig
  type t = About : { key       : 'a TUID.t;
                     json_desc : unit -> unit;
                     serialise : 'a PPX_Serialise.t;
                   } -> t
end

(* The register of types is indexed by (fully qualified) type names (strings) *)
module Register : sig
  val mem        : string -> bool
  val add        : string -> About.t -> unit
  val find_opt   : string -> About.t option
  val all        : unit -> string list (* List all registered types *)
end

(* Module to produce a description of types in JSON format *)
module JSONindex : sig

  val check_duplicates : bool ref

  (* Low-level primitives, for the Arsenal PPX; do not use yourself *)
  type t
  val mem  : string -> bool
  val find : string -> (string*JSON.t) list
  val mark : string -> t option
  val add  : t -> (string*JSON.t) list -> unit

  (* Functions to be used by user *)
  val populate : string -> unit (* Produces the JSON description from a given entry point *)
  (* Gives back the produced JSON description,
     "id" and "description" appearing in JSON header,
     "sentence_info" being the description of what is produced for each sentence, besides 1 cst. *)
  val out :
    id:string ->
    description:string ->
    sentence_info: (string*JSON.t) list ->
    toptype:string ->
    JSON.t
end
     

(**********************************************)
(* Built-in types (bool, int, list and option *)
(**********************************************)

val typestring_bool: unit -> string
val typestring_int : unit -> string
val typestring_list: (unit -> string) -> unit -> string
val typestring_option: (unit -> string) -> unit -> string

val json_desc_bool  : unit -> unit
val json_desc_int   : unit -> unit
val json_desc_list  : (unit -> string) -> unit -> unit
val json_desc_option: (unit -> string) -> unit -> unit

val serialise_bool   : bool PPX_Serialise.t
val serialise_int    : int PPX_Serialise.t
val serialise_list   : ('a PPX_Serialise.t) -> 'a list PPX_Serialise.t
val serialise_option : ('a PPX_Serialise.t) -> 'a option PPX_Serialise.t

val random_bool   : bool PPX_Random.t
val true_p        : float -> bool PPX_Random.t
val random_int    : int  PPX_Random.t
(* min by default is 0; empty is the probability of terminatig the list generation at every step, if unspecified empty=0.5 *)
val random_list   :
  ?min:int -> ?max:int -> ?empty:float -> ('b PPX_Random.t) -> 'b list PPX_Random.t
(* p is the probability of None *)
val random_option : ?p:float -> ('b PPX_Random.t) -> 'b option PPX_Random.t
(* Same as above, the float is the probability of None *)
val none_p : float -> ('b PPX_Random.t) -> 'b option PPX_Random.t

(* Other useful primitives for random generation *)
val random_ascii      : char PPX_Random.t          (* Random char *)
val random_ascii_pair : (char * char) PPX_Random.t (* Random pair of chars *)
val ( *! ) : ('b PPX_Random.t) -> ('c PPX_Random.t) -> ('b * 'c) PPX_Random.t (* Random product *)
val random_string : ?min:int -> ?max:int -> ?eos:float -> string PPX_Random.t (* Random string *)

type 'a distribution = ('a*float) list

(* Useful for normalizing *)
val flatten      : ('a -> 'a list option) -> 'a -> 'a list
val flatten_list : ('a -> 'a list option) -> 'a list -> 'a list


(*************************************************)
(* Small extension of the standard Result module *)
(*************************************************)

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
(***********************)

module Entity :
sig
  val one_entity_kind : bool ref
  val warnings : [ `NoSubst of string ] list ref
  val strict   : float ref
  val ppkind   : bool ref
  val show_substitutions : bool ref
  val init : unit -> unit

  type 'a t [@@deriving arsenal]
  val get_subst : 'a t -> string
  val pp    : 'a TUID.t -> 'a PPX_Serialise.t -> 'a t pp
  val to_id : string -> (string * int, string) result
  val entity_mk :
    string ->
    ?kind:'a option ->
    ?counter:int ->
    ?substitution:string option -> unit -> ('a t, 'b) Result.result
  val pick : ('a * float) list -> PPX_Random.state -> 'a
  val get_types : unit -> tag list
end

(******************)
(* Postprocessing *)
(******************)

type cst_process =
  ?global_options:(string * JSON.t) list ->
  ?options:(string * JSON.t) list ->
  ?original:string ->
  ?ep:string ->
  ?cleaned:string ->
  id:JSON.t ->
  to_sexp:(JSON.t -> Sexp.t) -> (* Turns 1 polish notation into a Sexp with substituted placeholders *)
  JSON.t -> (* Contents of ths "cst" field sent to the reformulator *)
  JSON.t

val postprocess : cst_process -> JSON.t -> JSON.t
