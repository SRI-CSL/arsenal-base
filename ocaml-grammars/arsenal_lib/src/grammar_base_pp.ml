open Arsenal_lib
open Grammar_base

(************************)
(* Pretty-print helpers *)

let sverb = Qualif.Verb{ vplural = Singular; neg = false; aux = None }
let pverb = Qualif.Verb{ vplural = Plural; neg = false; aux = None }

let sd_noun = Qualif.Noun{ article = Definite; singplu = Singular }
let si_noun = Qualif.Noun{ article = Indefinite; singplu = Singular }
let pd_noun = Qualif.Noun{ article = Definite; singplu = Plural }
let pi_noun = Qualif.Noun{ article = Indefinite; singplu = Plural }

let singplu_choose = !&[Singular; Plural]
let bchoose = !&[true; false]

let verb_mk () = Qualif.Verb{ vplural = singplu_choose (); neg = false; aux = None }
let noun_mk () = Qualif.Noun{ article = if bchoose () then Definite else Indefinite;
                              singplu = if bchoose () then Singular else Plural }

let rec conjugate state stem =
  let stem, rest =
    match String.split_on_char ' ' stem with
    | stem::rest -> stem, rest
    | [] -> failwith "conjugate: should not happen"
  in
  let Qualif.(Verb{ vplural; neg; aux }) = state in
  let l = String.length stem in
  let stem = 
    match aux with
    | Some `Need ->
       let aux =
         match vplural with
         | Plural   -> "need"
         | Singular -> "needs"
       in
       !![
           if neg then aux^" not to "^stem
           else aux^" to "^stem
         ]

    | Some `PresentPart ->
       let steming =
       
       (* "...ie" -> "...ying" *)
       if Str.(string_match (regexp "ie") stem (l-2)) 
          then String.sub stem 0 (l-1)^"ying"
       
       (* 
        todo: extend to general pattern: single vowel followed by single consonant
        should double the last consonant
       *)
       (* "...et" -> "...etting" *)
       else if Str.(string_match (regexp "et") stem (l-2)) 
          then stem^"ting"
       else if Str.(string_match (regexp "un") stem (l-2)) 
          then stem^"ning"
       else
          match String.sub stem (l - 1) 1 with
          | "e" -> String.sub stem 0 (l - 1)^"ing"
          | _ -> stem^"ing"
       in
       !![
           if neg then "not "^steming
           else steming
         ]

    | Some `PastPart ->
       let stemed =
        match stem with (* capture irregular forms here*)
        | "set" -> "set"
        | "get" -> "got"
        | "run" -> "ran"
        | "begin" -> "began"
        | _ ->
          match String.sub stem (l - 1) 1 with
          | "y" -> String.sub stem 0 (l - 1)^"ied"
          | "e" -> stem^"d"
          | _ -> stem^"ed"
       in
       !![
           if neg then "not "^stemed
           else stemed
         ]

    | Some a ->
       let a = match a with
         | `Can   -> "can"
         | `Shall -> "shall"
         | `Will  -> "will"
         | `May   -> "may"
         | `Might -> "might"
         | `Must  -> "must"
         | _  -> failwith "should not happen"
       in
       !![
           if neg then a^" not "^stem
           else a^" "^stem
         ]

    | None ->
       match stem with
       | "be"   -> let a = match vplural with Plural -> "are" | Singular -> "is" in
                   !![ if neg then a^" not" else a ]
       | "have" -> if neg then
                     !?[ F "%t %s" // conjugate state "do" // "have" ]
                   else
                     !![ match vplural with Plural -> "have" | Singular -> "has" ]

       | "do"   -> let a = match vplural with Plural -> "do" | Singular -> "does" in
                   !![ if neg then a^" not" else a ]
       | _      -> if neg then
                     !?[ F "%t %s" // conjugate state "do" // stem ]
                   else
                     !![ match vplural with
                         | Plural -> stem
                         | Singular -> 
                            match String.sub stem (l - 1) 1 with
                            | "y" -> String.sub stem 0 (l - 1)^"ies"
                            | _ -> stem^"s" ]
  in
  match rest with
  | [] -> stem
  | _::_ ->
     let rest = pp_list ~sep:" " return rest in
     !?[ F "%t %t" // stem // rest]

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

let pp_integer = Entity.pp key_integerT serialise_integerT

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
  | Between{lower_bound; upper_bound} -> !?[
      F "between %t and %t" // print_arg lower_bound // print_arg upper_bound;
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
  | Percent p -> F "%t percent of %t " // pp_range pp_integer p // the |> print
  | Fraction p -> F "%t of %t " // pp_range pp_fraction p // the |> print
    

let pp_proportion_option = function
  | None    -> F "the " |> print
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
  pp_range (fun (Time{number;unit}) -> F "%t %t" // pp_integer number // pp_arg unit |> print)
    
let pp_q_time (QT range) = pp_range_aux (pp_time_unit Plural) range
