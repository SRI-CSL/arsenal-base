open Format
open Sexplib
open Std

open Arsenal_lib

let concat_max = ref 4   (* Maximal length of concatenations *)
let string_max = ref 10  (* Maximal length of random strings *)

(***********)
(* Grammar *)

type kchar = Char [@@deriving arsenal, show { with_path = false }]
type kstring = String [@@deriving arsenal, show { with_path = false }]

type tchar = kchar Entity.t     [@@deriving arsenal]
type tstring = kstring Entity.t [@@deriving arsenal]

type terminal =
  | Specific of tstring [@weight 10]
  | Empty
  | CharacterRange of tchar*tchar [@weight 4]
  (* | Backslash | Slash | Dollar | Plus | Star | Hat | Return | Newline | Tab (\* escapes are in String *\) *)
  | Word | Any | Digit | Space | NotWord | NotDigit | NotSpace
[@@deriving arsenal]

type re = Terminal of terminal [@weight fun state -> 2. *. depth state ]
        | StartOfLine of re
        | EndOfLine of re
        | Plus of re
        | Star of re
        | Or of re*re
        | Concat of (re list[@random random_list ~min:2 random_re])
[@@deriving arsenal]
