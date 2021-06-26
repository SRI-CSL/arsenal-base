open Arsenal_lib
open Grammar_base
open Qualif

(*******************************)
(* Elements of English grammar *)

(* Different modes for verbs and nouns *)
val sverb : verb
val pverb : verb
val sd_noun : noun (* singular definite *)
val si_noun : noun (* singular indefinite *)
val pd_noun : noun (* plural definite *)
val pi_noun : noun (* plural indefinite *)
val bchoose : unit -> bool (* Choose a Boolean *)
val verb_mk : unit -> verb (* Make random verb mode *)
val noun_mk : unit -> noun (* Make random noun mode *)
val conjugate : verb -> string -> print
(* prints random modal verb, possibly none *)
val modal_pure : print
(* prints verb with modal verb possibly added *)
val modal : verb -> string -> print

val agree    : singplu -> string -> string (* prints s if plural *)
val bnot     : bool -> string (* adds " not" if condition is negative *)
val bno      : bool -> string (* adds " no" if condition is negative *)
val the      : print         (* the or empty string*)
val that     : print         (* that or empty string*)
val preposition : print      (* random preposition *)
val suchas   : print         (* such as and synonyms *)
val sep_last : unit -> string (* , / and, with or without Oxford comma *)
val incaseof : bool -> print (* synonyms of "in case of" / "in absence of" *)
val incase   : print  (* when / whenever / in case *)
val further  : print  (* further / more *)
val colon    : print  (* colon or not *)
val needed   : print  (* needed / necessary /... *)
val every    : print  (* each, every, etc *)
 
(*********************************************************)
(* Pretty-printers for the types defined in Base_grammar *)

val pp_qualif     : (singplu -> 'a pp) -> 'a qualif pp
val pp_integer    : integerT Entity.t pp
val pp_range      : ('a pp) -> 'a range pp
val pp_fraction   : fraction pp
val pp_proportion : proportion pp
val pp_proportion_option : proportion option pp
val pp_time_unit  : singplu -> time_unit pp
val pp_range_aux  : (time_unit pp) -> time range pp
val pp_q_time     : q_time pp
