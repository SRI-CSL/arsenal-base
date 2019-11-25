open Format
open Sexplib.Std

open Arsenal_lib
open Base_grammar

(************************)
(* Pretty-print helpers *)

(* Easy choose *)
(* print string *)
let (>>) s fmt  = fprintf fmt "%s" s
(* For lists of strings *)
let (!!)  l fmt = choose(List.map (fun s -> lazy (s>>fmt), 1) l)
let (!!!) l fmt = choose(List.map (fun (s,i) -> lazy (s>>fmt),i) l)
(* For lists of %t functions *)
let (!~)  l fmt = choose(List.map (fun s -> lazy(s fmt),1) l)
let (!~~) l fmt = choose(List.map (fun (s,i) -> lazy(s fmt),i) l)
(* For lists of 'a *)
let (!&)  l ()  = choose(List.map (fun s -> lazy s,1) l)
let (!&&)  l () = choose(List.map (fun (s,i) -> lazy s,i) l)
(* For lists of lazy things *)
let (!?)  l  = choose(List.map (fun s -> s,1) l)
let (!??) l  = choose l

(* Easy weights *)
let (?~) b   = if b then 1 else 0
let (~?) b   = if b then 0 else 1
let (??~) o  = match o with Some _ -> 1 | None -> 0
let (~??) o  = 1 - ??~o
let (++) w1 w2 = 1 - (w1 * w2)

(* Easy extension of pp function to option type *)
let (?+) pp_arg fmt = function Some x -> pp_arg fmt x | None -> !![""] fmt
let (?++) = function Some x -> true | None -> false

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

let pos_be state fmt = !?[
    lazy (fprintf fmt "%s" (is state));
    lazy (fprintf fmt "should be");
    lazy (fprintf fmt "must be");
    lazy (fprintf fmt "need%s to be" (thirdp state));
  ]

let neg_be state fmt = !?[
    lazy (fprintf fmt "%s not" (is state));
    lazy (fprintf fmt "should not be");
    lazy (fprintf fmt "must not be");
    lazy (fprintf fmt "need%s not be" (thirdp state));
    lazy (fprintf fmt "%s not need to be" (does state));
  ]

let pos_have state fmt = !?[
    lazy (fprintf fmt "%s" (has state));
    lazy (fprintf fmt "suffer%s from" (thirdp state));
  ]

let neg_have state fmt = !?[
    lazy (fprintf fmt "%s not have" (does state));
    lazy (fprintf fmt "%s no" (has state));
    lazy (fprintf fmt "%s not suffer from" (does state));
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

let pp_integer = Entity.pp (fun fmt `Integer -> "Integer" >> fmt)

(**************)
(* Quantities *)

let pp_range print_arg fmt = function
  | Exact q        -> !?[lazy (fprintf fmt "%a" print_arg q)]
  | MoreThan q     -> !?[
      lazy (fprintf fmt "more than %a" print_arg q);
      lazy (fprintf fmt "at least %a" print_arg q);
      lazy (fprintf fmt "over %a" print_arg q);
      lazy (fprintf fmt "%a or more" print_arg q);
    ]
  | LessThan q     -> !?[
      lazy (fprintf fmt "less than %a" print_arg q);
      lazy (fprintf fmt "at most %a" print_arg q);
      lazy (fprintf fmt "under %a" print_arg q);
      lazy (fprintf fmt "below %a" print_arg q);
      lazy (fprintf fmt "%a or less" print_arg q);
      lazy (fprintf fmt "up to %a" print_arg q);
    ]
  | Approximately q -> !??[
      lazy (fprintf fmt "approximately %a" print_arg q),3;
      lazy (fprintf fmt "about %a" print_arg q),3;
      lazy (fprintf fmt "roughly %a" print_arg q),1;
    ]
  | Between(q1,q2) -> !?[
      lazy (fprintf fmt "between %a and %a" print_arg q1 print_arg q2);
    ]

let pp_fraction fmt = function
  | Half -> !!["half"] fmt
  | Third -> !!["third"] fmt
  | Quarter -> !!["quarter"] fmt
    
let pp_proportion fmt q =
  let s = !!![" of the",1; "",2] in
  match q with
  | All    -> fprintf fmt "all%t " s
  | NoneOf -> fprintf fmt "none of the "
  | Most   -> fprintf fmt "most%t " s
  | Few    -> fprintf fmt "few%t " s
  | Many   -> !?[ lazy(fprintf fmt "many%t " s); lazy(fprintf fmt "numerous%t " s)]
  | Several-> fprintf fmt "several "
  | Range p   -> fprintf fmt "%a " (pp_range pp_integer) p
  | Percent p -> fprintf fmt "%a percent of %t" (pp_range pp_integer) p the
  | Fraction p -> fprintf fmt "%a of %t" (pp_range pp_fraction) p the
    

let pp_proportion_option fmt = function
  | None    -> fprintf fmt "%t" (article pd_noun)
  | Some pe -> fprintf fmt "%a" pp_proportion pe

let pp_time_unit state fmt = function
  | Second -> fprintf fmt "second%s" (nplural state)
  | Minute -> fprintf fmt "minute%s" (nplural state)
  | Hour -> fprintf fmt "hour%s" (nplural state)
  | Day  -> fprintf fmt "day%s" (nplural state)
  | Week -> fprintf fmt "week%s" (nplural state)
  | Month -> fprintf fmt "month%s" (nplural state)
  | Year  -> fprintf fmt "year%s" (nplural state)

let pp_range_aux pp_arg =
  pp_range (fun fmt (Time(i,a)) -> fprintf fmt "%a %a" pp_integer i pp_arg a)
    
let pp_q_time fmt (QT range) = pp_range_aux (pp_time_unit pd_noun) fmt range
