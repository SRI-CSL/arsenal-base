open Sexplib
open Arsenal_lib

(*************)
(* Terminals *)

type integer  = [`Integer] Entity.t  [@@deriving arsenal]

(**************)
(* Qualifying *)

type article =
  | Definite   [@weight 4]
  | Indefinite [@weight 2]
  | All [@weight 0]
  | Som [@weight 0]
[@@deriving arsenal]

type singplu = Singular | Plural
                            [@@deriving arsenal]

module Qualif = struct

  type verb = Verb of {
      vplural : bool;
      (* person  : [`First | `Second | `Third ]; *)
      (* tense   : [ `Present | `Infinitive ]; *)
      neg     : bool;
      aux     : [`Can | `Shall | `Will | `May | `Might | `Must | `Need ] option
    }

  type noun = Noun of {
      article : article;
      singplu : singplu
    } [@@deriving arsenal]

end
              
type 'a qualif = Qualif of {
      noun : 'a;
      qualif : Qualif.noun;
    }[@@deriving arsenal]


(**************)
(* Quantities *)

              
type 'q range = Exact of 'q [@weight 5]
      | MoreThan of 'q      [@weight 4]
      | LessThan of 'q      [@weight 4]
      | Approximately of 'q [@weight 3]
      | Between of 'q*'q    [@weight 2]
[@@deriving arsenal]

type fraction = Half | Third | Quarter
[@@deriving arsenal]
                
type proportion =
  | All
  | NoneOf
  | Most
  | Few
  | Several
  | Many
  | Range of integer range
  | Percent of integer range
  | Fraction of fraction range
[@@deriving arsenal]

type time_unit = Second | Minute | Hour | Day | Week | Month | Year
[@@deriving show { with_path = false }, arsenal]

type time = Time of integer * time_unit [@@deriving arsenal]

type q_time  = QT of time range [@@deriving arsenal]
