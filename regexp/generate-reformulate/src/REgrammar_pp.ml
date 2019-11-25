open Format
open Sexplib
open Std

open Arsenal_lib
open Base_grammar_pp
open REgrammar

(***************)
(* NL Printing *)

let pp_tchar = Entity.pp pp_kchar
let pp_tstring = Entity.pp pp_kstring

let pp_terminal fmt = function
  | Specific s -> !?[
      lazy (fprintf fmt "the string %a" pp_tstring s);
      lazy (fprintf fmt "string %a" pp_tstring s);
    ]
  | Empty    -> !?[
      lazy (fprintf fmt "an empty string");
      lazy (fprintf fmt "the empty string");
      lazy (fprintf fmt "an empty word");
      lazy (fprintf fmt "the empty word");
    ]
  | CharacterRange(a,z) -> !?[
      lazy (fprintf fmt "a character between %a and %a" pp_tchar a pp_tchar z);
      lazy (fprintf fmt "any character between %a and %a" pp_tchar a pp_tchar z);
    ]
  | Word     -> !?[
      lazy (fprintf fmt "a word character");
      lazy (fprintf fmt "any word character");
    ]
  | Any      -> !?[
      lazy (fprintf fmt "a character");
      lazy (fprintf fmt "any character");
    ]
  | Digit    -> !?[
      lazy (fprintf fmt "a digit");
      lazy (fprintf fmt "any digit");
    ]
  | Space    -> !?[
      lazy (fprintf fmt "a space character");
      lazy (fprintf fmt "any space character");
    ]
  | NotWord  -> !?[
      lazy (fprintf fmt "a non-word character");
      lazy (fprintf fmt "any non-word character");
    ]
  | NotDigit -> !?[
      lazy (fprintf fmt "a non-digit character");
      lazy (fprintf fmt "any non-digit character");
    ] 
  | NotSpace -> !?[
      lazy (fprintf fmt "a non-space character");
      lazy (fprintf fmt "any non-space character");
    ]

let rec pp_list pp_arg fmt = function
  | []   -> !?[lazy (fprintf fmt "")]
  | [a]  -> !?[lazy (fprintf fmt "%a" pp_arg a)]
  | c::l -> !?[
      lazy (fprintf fmt "%a, followed by %a" pp_arg c (pp_list pp_arg) l);
      lazy (fprintf fmt "%a, and then %a" pp_arg c (pp_list pp_arg) l);
    ]
    
let rec pp_re fmt = function
  | Terminal t    -> !?[lazy (pp_terminal fmt t)]
  | StartOfLine r -> !?[
      lazy (fprintf fmt "line beginning with %a" pp_re r);
      lazy (fprintf fmt "line starting with %a" pp_re r);
    ]
  | EndOfLine r   -> !?[lazy (fprintf fmt "line ending with %a" pp_re r);]
  | Plus r        -> !?[
      lazy (fprintf fmt "one or more repetitions of %a" pp_re r);
      lazy (fprintf fmt "at least one repetition of %a" pp_re r);
    ]
  | Star r        -> !?[
      lazy (fprintf fmt "zero or more repetitions of %a" pp_re r);
      lazy (fprintf fmt "any number of repetitions of %a" pp_re r);
    ]
  | Or(r1,r2)     -> !?[
      lazy (fprintf fmt "either %a or %a" pp_re r1 pp_re r2);
      lazy (fprintf fmt "%a or %a" pp_re r1 pp_re r2);
    ]
  | Concat l      -> !?[lazy (pp_list pp_re fmt l)]
