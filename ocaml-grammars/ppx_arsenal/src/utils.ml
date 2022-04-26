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

let raise_errorf = Ppx_deriving.raise_errorf

let ident prefix typestr =
  Exp.ident (mknoloc (Ppx_deriving.mangle_lid (`Prefix prefix) typestr))

let ident_decl prefix type_decl =
  evar (Ppx_deriving.mangle_type_decl (`Prefix prefix) type_decl)

let efst loc x = [%expr fst [%e x ]]
let esnd loc x = [%expr snd [%e x ]]

let hash_list loc ~init elts =
  let aux (f_sofar, e_sofar) (f,e) =
    [%expr CCHash.pair [%e f_sofar] [%e f]], [%expr [%e e_sofar], [%e e]]
  in
  elts |> List.fold_left aux init

let json_list loc ?(init=[%expr []]) elts =
  let aux sofar arg =  [%expr PPX_Serialise.json_cons [%e arg] [%e sofar]] in
  elts |> List.rev |> List.fold_left aux init

let list loc ?(init=[%expr []]) elts =
  let aux sofar arg =  [%expr [%e arg] :: [%e sofar]] in
  elts |> List.rev |> List.fold_left aux init

let list_pat loc ?(init=[%pat? []]) elts =
  let aux sofar arg =  [%pat? [%p arg] :: [%p sofar]] in
  elts |> List.rev |> List.fold_left aux init

let default_case typestring_expr loc =
  Exp.case [%pat? sexp ] [%expr PPX_Serialise.sexp_throw ~who:([%e typestring_expr]^".of_sexp") sexp ]

let attribute ~deriver s attrs =
  attrs
  |> Ppx_deriving.attr ~deriver s
  |> Ppx_deriving.Arg.get_flag ~deriver

let get_param param =
  "poly_"^param.txt
  |> Lexing.from_string
  (* |> Parse.longident
   * |> mknoloc
   * |> Exp.ident *)
  |> Parse.expression

let get_params type_decl =
  let aux param sofar = get_param param :: sofar in
  Ppx_deriving.fold_right_type_decl aux type_decl [] |> List.rev

let application_str loc args =
  let aux param sofar = [%expr PPX_Serialise.str_apply [%e sofar] [%e param]] in
  List.fold_right aux args

let with_path = ref None

let qualify loc exp path str =
  let path = path |> List.map str2exp |> list loc in
  let str  = str2exp str in
  match !with_path with
  | None      -> [%expr ![%e exp]                 ~path:[%e path] [%e str]]
  | Some mode -> [%expr ![%e exp] ~mode:[%e mode] ~path:[%e path] [%e str]]

let type_qualify        loc = qualify loc [%expr PPX_Serialise.type_qualify]
let constructor_qualify loc = qualify loc [%expr PPX_Serialise.constructor_qualify]

let is_option_type core_type =
  match core_type with
  | {ptyp_desc = Ptyp_constr ({txt = Lident "option" ; _}, [_]) ; _} -> true
  | _ -> false

let get_main_arg typs =
  let rec aux i = function
    | [] -> None
    | typ::_ when not(is_option_type typ) -> Some i
    | _::tail -> aux (i+1) tail
  in
  typs |> aux 0
