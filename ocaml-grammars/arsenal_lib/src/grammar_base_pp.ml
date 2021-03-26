open Arsenal_lib
open Grammar_base

(************************)
(* Pretty-print helpers *)

let sverb = Qualif.Verb{ vplural = false; neg = false; aux = None }
let pverb = Qualif.Verb{ vplural = true; neg = false; aux = None }

let sd_noun = Qualif.Noun{ article = Definite; singplu = Singular }
let si_noun = Qualif.Noun{ article = Indefinite; singplu = Singular }
let pd_noun = Qualif.Noun{ article = Definite; singplu = Plural }
let pi_noun = Qualif.Noun{ article = Indefinite; singplu = Plural }

let bchoose = !&[true; false]

let verb_mk () = Qualif.Verb{ vplural = bchoose (); neg = false; aux = None }
let noun_mk () = Qualif.Noun{ article = if bchoose () then Definite else Indefinite;
                              singplu = if bchoose () then Singular else Plural }

let rec conjugate state stem =
  let Qualif.(Verb{ vplural; neg; aux }) = state in
  match aux with
  | Some `Need ->
     !![
         if neg then "need not "^stem
         else if vplural then "need to "^stem
         else "needs to "^stem
       ]

  | Some a ->
     let a = match a with
       | `Can   -> "can"
       | `Shall -> "shall"
       | `Will  -> "will"
       | `May   -> "may"
       | `Might -> "might"
       | `Must  -> "must"
       | `Need  -> "need"
     in
     !![
         if neg then a^" not "^stem
         else a^" "^stem
       ]

  | None ->
     match stem with
     | "be"   -> let a = if vplural then "are" else "is" in
                 !![ if neg then a^" not" else a ]
     | "have" -> if neg then
                   !?[ F "%t %s" // conjugate state "do" // "have" ]
                 else
                   !![ if vplural then "have" else "has" ]

     | "do"   -> let a = if vplural then "do" else "does" in
                 !![ if neg then a else a^" not" ]
     | _      -> if neg then
                   !?[ F "%t %s" // conjugate state "do" // stem ]
                 else
                   !![ if vplural then stem
                       else
                         let l = String.length stem in
                         match String.sub stem (l - 1) 1 with
                         | "y" -> String.sub stem 0 (l - 1)^"ies"
                         | _ -> stem^"s" ]

let modal_pure = !!!["can ",1; "may ",1; "might ",1; "",4]
    
let modal state verb =
  let aux s = !??[ F "can %s"   // verb,1;
                   F "may %s"   // verb,1;
                   F "might %s" // verb,1;
                   F "%t" // s,4] in
  aux (conjugate state verb)

let agree singplu s =
  match singplu with
  | Singular -> s
  | Plural   -> s^"s"

let bnot b = if b then "" else " not"
let bno b = if b then "" else " no"

let that        = !!["that "; ""]
let preposition = !!["in"; "with"; "for"]
let suchas      = !!!["such as",2; "like",1; "including",2]

let sep_last = !&&[
    (" and"),3;
    (", and"),1; (* with Oxford comma *)
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

let pp_integer = Entity.pp serialise_integerT

(**************)
(* Qualifiers *)

let every = !!["every"; "each"; "any"]

let pp_qualif pp_arg (Qualif{ noun; qualif }) =
  let Qualif.(Noun{article; singplu}) = qualif in
  match article, singplu with
  | Definite, Singular   ->
     let art = !!![ "the",3; "this", 1; "that", 1] in
     !?[ F "%t %t" // art // pp_arg singplu noun ]
  | Definite, Plural   ->
     let art = !!![ "the",3; "these", 1; "those", 1] in
     !?[ F "%t %t" // art // pp_arg singplu noun ]
  | Indefinite, Singular ->
     !?[ F "a %t" // pp_arg singplu noun ]
  | Indefinite, Plural   ->
     pp_arg singplu noun
  | All, Singular   ->
     !?[ F "%t %t" // every // pp_arg singplu noun ]
  | All, Plural     ->
     !?[ F "all %t" // pp_arg singplu noun ]
  | Som, _   ->
     !?[ F "some %t" // pp_arg singplu noun ]
    

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

let the = !!!["the", 3; "",1]

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
  | None    -> F "the" |> print
  | Some pe -> F "%t" // pp_proportion pe |> print

let pp_time_unit state = function
  | Second -> !![ agree state "second" ]
  | Minute -> !![ agree state "minute" ]
  | Hour   -> !![ agree state "hour" ]
  | Day    -> !![ agree state "day" ]
  | Week   -> !![ agree state "week" ]
  | Month  -> !![ agree state "month" ]
  | Year   -> !![ agree state "year" ]

let pp_range_aux pp_arg =
  pp_range (fun (Time(i,a)) -> F "%t %t" // pp_integer i // pp_arg a |> print)
    
let pp_q_time (QT range) = pp_range_aux (pp_time_unit Plural) range
