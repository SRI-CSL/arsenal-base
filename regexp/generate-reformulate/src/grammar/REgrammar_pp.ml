open Format
open Sexplib
open Std

open Arsenal
open Arsenal_lib
open Grammar_base_pp
open REgrammar

(***************)
(* NL Printing *)

let pp_tchar   = Entity.pp key_kchar serialise_kchar
let pp_tstring = Entity.pp key_kstring serialise_kstring

let pp_terminal = function
  | Specific s ->
     let s = pp_tstring s in
     !?[
      F "the string %t" // s;
      F "string %t"     // s;
    ]
  | Empty    -> !?[
      F "an empty string";
      F "the empty string";
      F "an empty word";
      F "the empty word";
    ]
  | CharacterRange(a,z) ->
     let a = pp_tchar a in
     let z = pp_tchar z in
     !?[
      F "a character between %t and %t"   // a // z;
      F "any character between %t and %t" // a // z;
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
  | [a]  -> let a = pp_arg a in !?[F "%t" // a]
  | c::l ->
     let c = pp_arg c in
     let l = pp_list pp_arg l in
     !?[
      F "%t, followed by %t" // c // l;
      F "%t, and then %t"    // c // l;
    ]
    
let rec pp_re = function
  | Terminal t    -> pp_terminal t
  | StartOfLine r ->
     let r = pp_re r in
     !?[
      F "line beginning with %t" // r;
      F "line starting with %t"  // r;
       ]
  | EndOfLine r   ->
     let r = pp_re r in
     !?[F "line ending with %t" // r;]
  | Plus r        ->
     let r = pp_re r in
     !?[
         F "one or more repetitions of %t" // r;
         F "at least one repetition of %t" // r;
       ]
  | Star r        ->
     let r = pp_re r in
     !?[
         F "zero or more repetitions of %t" // r;
         F "any number of repetitions of %t" // r;
         F "any repetitions of %t" // r;
       ]
  | Or(r1,r2)     ->
     let r1 = pp_re r1 in
     let r2 = pp_re r2 in
     !?[
      F "either %t or %t" // r1 // r2;
      F "%t or %t"        // r1 // r2;
       ]
  | Concat l      -> pp_list pp_re l

let () = TUID.get_pp key_re := pp_re

let load = ()
