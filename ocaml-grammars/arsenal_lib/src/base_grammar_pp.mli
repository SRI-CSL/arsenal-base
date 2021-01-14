open Arsenal_lib
open Base_grammar

(*******************************)
(* Elements of English grammar *)

(* Different modes for verbs and nouns *)
type verb = { vplural : bool; }
type noun = { nplural : bool; definite : bool; noarticle : bool; }
val sverb : verb
val pverb : verb
val sd_noun : noun (* singular definite *)
val si_noun : noun (* singular indefinite *)
val pd_noun : noun (* plural definite *)
val pi_noun : noun (* plural indefinite *)
val bchoose : unit -> bool (* Choose a Boolean *)
val verb_mk : unit -> verb (* Make random verb mode *)
val noun_mk : unit -> noun (* Make random noun mode *)
val thirdp  : verb -> string (* prints s if in third person singular *)
val does    : verb -> string (* conjugate to do according to verb mode *)
val is      : verb -> string (* conjugate to be according to verb mode *)
val has     : verb -> string (* conjugate to have according to verb mode *)

(* prints random modal verb, possibly none *)
val modal_pure : print
(* prints verb with modal verb possibly added *)
val modal : verb -> string -> print

val nplural  : noun -> string (* prints s if plural *)
val bnot     : bool -> string (* adds " not" if condition is negative *)
val bno      : bool -> string (* adds " no" if condition is negative *)
val pos_be   : verb -> print (* to be with possibly modal *)
val neg_be   : verb -> print (* not to be with possibly modal *)
val pos_have : verb -> print (* to have / suffer with possibly modal *)
val neg_have : verb -> print (* not to have / suffer with possibly modal *)
val the      : print         (* the or empty string*)
val article  : noun -> print (* article, agreeing with the noun mode *)
val that     : print         (* that or empty string*)
val preposition : print      (* random preposition *)
val suchas   : print         (* such as and synonyms *)
val sep_last : unit -> string * string (* , / and, with or without Oxford comma *)
val incaseof : bool -> print (* synonyms of "in case of" / "in absence of" *)
val incase   : print  (* when / whenever / in case *)
val further  : print  (* further / more *)
val colon    : print  (* colon or not *)
val needed   : print  (* needed / necessary /... *)

(*********************************************************)
(* Pretty-printers for the types defined in Base_grammar *)

val pp_integer    : [ `Integer ] Entity.t pp
val pp_range      : ('a pp) -> 'a range pp
val pp_fraction   : fraction pp
val pp_proportion : proportion pp
val pp_proportion_option : proportion option pp
val pp_time_unit  : noun -> time_unit pp
val pp_range_aux  : (time_unit pp) -> time range pp
val pp_q_time     : q_time pp
