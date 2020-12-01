open Format
open Sexplib.Std

open Arsenal_lib
open Base_grammar

(************************)
(* Pretty-print helpers *)

type verb = {
  vplural : bool;
  (* person  : [`First | `Second | `Third ];
   * tense   : [ `Present ] *)
}

type noun = {
  nplural  : bool;
  definite : bool;
  noarticle: bool
}

let sverb = { vplural = false }
let pverb = { vplural = true }
let sd_noun = { nplural = false; definite = true; noarticle=false }
let si_noun = { nplural = false; definite = false; noarticle=false }
let pd_noun = { nplural = true; definite = true; noarticle=false }
let pi_noun = { nplural = true; definite = false; noarticle=false }
let bchoose = !&[true; false]
let verb_mk () = { vplural = bchoose () }
let noun_mk () = { nplural = bchoose (); definite = bchoose(); noarticle=false }

let thirdp state = if state.vplural then "" else "s"
let does state   = if state.vplural then "do" else "does"
let is state     = if state.vplural then "are" else "is"
let has state    = if state.vplural then "have" else "has"

let modal_pure = !!!["can ",1; "may ",1; "might ",1; "",4]
    
let modal state verb =
  let aux s = !!!["can "^verb,1; "may "^verb,1; "might "^verb,1; s,4] in
  match verb with
  | "be"   -> aux (is state)
  | "have" -> aux (has state)
  | "do"   -> aux (does state)
  | _      -> aux (verb^(thirdp state))

let nplural state = if state.nplural then "s" else ""
  
let bnot b = if b then "" else " not"
let bno b = if b then "" else " no"

let pos_be state = !?[
    F "%s" // is state;
    F "should be";
    F "must be";
    F "need%s to be" // thirdp state;
  ]

let neg_be state = !?[
    F "%s not" // is state;
    F "should not be";
    F "must not be";
    F "need%s not be" // thirdp state;
    F "%s not need to be" // does state;
  ]

let pos_have state = !?[
    F "%s" // has state;
    F "suffer%s from" // thirdp state;
  ]

let neg_have state = !?[
    F "%s not have" // does state;
    F "%s no" // has state;
    F "%s not suffer from" // does state;
  ]

let the = !!["the ";""]

let article state = !!![
    "the ", ?~(state.definite) * ~?(state.noarticle);
    "a ", ~?(state.nplural) * ~?(state.definite) * ~?(state.noarticle);
    "", ?~(state.nplural)
  ]

let that        = !!["that "; ""]
let preposition = !!["in"; "with"; "for"]
let suchas      = !!!["such as",2; "like",1; "including",2]

let sep_last = !&&[
    (","," and"),3;
    (",",", and"),1; (* with Oxford comma *)
  ]

let incaseof b =
  if b then !!["in case of"; "in presence of"]
  else !!["in absence of"]

let incase    = !!["when"; "whenever"; "in case"]
let further   = !!["further"; "more"]
let colon     = !!![":",1; "",2]
let needed    = !!["needed"; "necessary"; "mandatory"; "required"]

(*************)
(* Terminals *)

let pp_integer = Entity.pp (fun fmt `Integer -> return "Integer" fmt)

(**************)
(* Quantities *)

let pp_range print_arg = function
  | Exact q        -> !?[F "%t" // print_arg q]
  | MoreThan q     -> !?[
      F "more than %t" // print_arg q;
      F "at least %t" // print_arg q;
      F "over %t" // print_arg q;
      F "%t or more" // print_arg q;
    ]
  | LessThan q     -> !?[
      F "less than %t" // print_arg q;
      F "at most %t" // print_arg q;
      F "under %t" // print_arg q;
      F "below %t" // print_arg q;
      F "%t or less" // print_arg q;
      F "up to %t" // print_arg q;
    ]
  | Approximately q -> !??[
      F "approximately %t" // print_arg q,3;
      F "about %t" // print_arg q,3;
      F "roughly %t" // print_arg q,1;
    ]
  | Between(q1,q2) -> !?[
      F "between %t and %t" // print_arg q1 // print_arg q2;
    ]

let pp_fraction = function
  | Half    -> !!["half"]
  | Third   -> !!["third"]
  | Quarter -> !!["quarter"]
    
let pp_proportion q =
  let s = !!![" of the",1; "",2] in
  match q with
  | All    -> F "all%t " // s  |> print
  | NoneOf -> F "none of the " |> print
  | Most   -> F "most%t " // s |> print
  | Few    -> F "few%t " // s  |> print
  | Many   -> !?[ F "many%t " // s; F "numerous%t " // s]
  | Several-> F "several " |> print
  | Range p   -> F "%t " // pp_range pp_integer p |> print
  | Percent p -> F "%t percent of %t" // pp_range pp_integer p // the |> print
  | Fraction p -> F "%t of %t" // pp_range pp_fraction p // the |> print
    

let pp_proportion_option = function
  | None    -> F "%t" // article pd_noun  |> print
  | Some pe -> F "%t" // pp_proportion pe |> print

let pp_time_unit state = function
  | Second -> F "second%s" // nplural state |> print
  | Minute -> F "minute%s" // nplural state |> print
  | Hour   -> F "hour%s"   // nplural state |> print
  | Day    -> F "day%s"    // nplural state |> print
  | Week   -> F "week%s"   // nplural state |> print
  | Month  -> F "month%s"  // nplural state |> print
  | Year   -> F "year%s"   // nplural state |> print

let pp_range_aux pp_arg =
  pp_range (fun (Time(i,a)) -> F "%t %t" // pp_integer i // pp_arg a |> print)
    
let pp_q_time (QT range) = pp_range_aux (pp_time_unit pd_noun) range
