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

let pp_terminal = function
  | Specific s -> !?[
      F "the string %t" // pp_tstring s;
      F "string %t" // pp_tstring s;
    ]
  | Empty    -> !?[
      F "an empty string";
      F "the empty string";
      F "an empty word";
      F "the empty word";
    ]
  | CharacterRange(a,z) -> !?[
      F "a character between %t and %t" // pp_tchar a // pp_tchar z;
      F "any character between %t and %t" // pp_tchar a // pp_tchar z;
    ]
  | Word     -> !?[
      F "a word character";
      F "any word character";
    ]
  | Any      -> !?[
      F "a character";
      F "any character";
    ]
  | Digit    -> !?[
      F "a digit";
      F "any digit";
    ]
  | Space    -> !?[
      F "a space character";
      F "any space character";
    ]
  | NotWord  -> !?[
      F "a non-word character";
      F "any non-word character";
    ]
  | NotDigit -> !?[
      F "a non-digit character";
      F "any non-digit character";
    ] 
  | NotSpace -> !?[
      F "a non-space character";
      F "any non-space character";
    ]

let rec pp_list pp_arg = function
  | []   -> !?[F ""]
  | [a]  -> !?[F "%t" // pp_arg a]
  | c::l -> !?[
      F "%t, followed by %t" // pp_arg c // pp_list pp_arg l;
      F "%t, and then %t" // pp_arg c // pp_list pp_arg l;
    ]
    
let rec pp_re = function
  | Terminal t    -> pp_terminal t
  | StartOfLine r -> !?[
      F "line beginning with %t" // pp_re r;
      F "line starting with %t" // pp_re r;
    ]
  | EndOfLine r   -> !?[F "line ending with %t" // pp_re r;]
  | Plus r        -> !?[
      F "one or more repetitions of %t" // pp_re r;
      F "at least one repetition of %t" // pp_re r;
    ]
  | Star r        -> !?[
      F "zero or more repetitions of %t" // pp_re r;
      F "any number of repetitions of %t" // pp_re r;
    ]
  | Or(r1,r2)     -> !?[
      F "either %t or %t" // pp_re r1 // pp_re r2;
      F "%t or %t" // pp_re r1 // pp_re r2;
    ]
  | Concat l      -> pp_list pp_re l
