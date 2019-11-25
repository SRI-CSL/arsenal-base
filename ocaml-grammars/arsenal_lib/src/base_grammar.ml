open Sexplib
open Sexplib.Std

open Arsenal_lib

(*************)
(* Terminals *)

type integer  = [`Integer] Entity.t  [@@deriving arsenal]

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
