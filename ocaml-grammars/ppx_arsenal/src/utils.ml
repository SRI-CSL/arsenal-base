(* Copyright (C) 2019 Stephane.Graham-Lengrand <disteph@gmail.com> (2019)
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version, with the OCaml static compilation exception.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library.  If not, see <http://www.gnu.org/licenses/>.
 *)

open Ppxlib
open Ast_helper
open Ppx_deriving.Ast_convenience

let str2exp x = x |> Const.string |> Exp.constant
let int2exp x = x |> Const.int |> Exp.constant

let with_path = ref true
let fully_qualified path str =
  if !with_path then
    let aux sofar modul = sofar ^ modul ^ "/" in
    (List.fold_left aux "" path)^str
  else
    str

let raise_errorf = Ppx_deriving.raise_errorf

let ident prefix typestr =
  Exp.ident (mknoloc (Ppx_deriving.mangle_lid (`Prefix prefix) typestr))

let ident_decl prefix type_decl =
  Exp.ident (mknoloc (Lident (Ppx_deriving.mangle_type_decl (`Prefix prefix) type_decl)))

let efst loc x = [%expr fst [%e x ]]
let esnd loc x = [%expr snd [%e x ]]

let hash_list loc ~init elts =
  let aux (f_sofar, e_sofar) (f,e) =
    [%expr CCHash.pair [%e f_sofar] [%e f]], [%expr [%e e_sofar], [%e e]]
  in
  elts |> List.rev |> List.fold_left aux init

let json_list loc ?(init=[%expr []]) elts =
  let aux sofar arg =  [%expr PPX_Serialise.json_cons [%e arg] [%e sofar]] in
  elts |> List.rev |> List.fold_left aux init

let list loc ?(init=[%expr []]) elts =
  let aux sofar arg =  [%expr [%e arg] :: [%e sofar]] in
  elts |> List.rev |> List.fold_left aux init

let list_pat loc ?(init=[%pat? []]) elts =
  let aux sofar arg =  [%pat? [%p arg] :: [%p sofar]] in
  elts |> List.rev |> List.fold_left aux init

let default_case loc = Exp.case [%pat? sexp ] [%expr PPX_Serialise.sexp_throw sexp ]
