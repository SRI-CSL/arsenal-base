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

open Utils
   
let deriver_name = "typestring"

(* Parse Tree and PPX Helpers *)

let parse_options loc l =
  with_path := None;  (* Use runtime option for determining how to qualify paths *)
  let aux (name, pexp) = 
    match name with
    | "with_path"  ->
       let expr =
         try
           if
             Ppx_deriving.Arg.(get_expr ~deriver:deriver_name bool) pexp
           then
             [%expr Some 0]
           else
             [%expr None]
         with
           _ -> pexp
       in
       with_path := Some expr
    | _ ->
       raise_errorf ~loc:pexp.pexp_loc
         "The %s deriver takes no option %s." deriver_name name
  in
  List.iter aux l

(* Generator Type *)

let typestring_type_of_decl loc ~options ~path:_path type_decl =
  parse_options loc options;
  Ppx_deriving.poly_arrow_of_type_decl
    (fun _var -> [%type: unit -> string])
    type_decl
    [%type: unit -> string]

(* Signature and Structure Components *)

let sig_of_type ~options ~path type_decl =
  let loc = type_decl.ptype_loc in
  parse_options loc options;
  [Sig.value
     (Val.mk
        (mknoloc (Ppx_deriving.mangle_type_decl (`Prefix "typestring") type_decl))
        (typestring_type_of_decl loc ~options ~path type_decl))]

let str_of_type ~options ~path type_decl =
  let loc = type_decl.ptype_loc in
  parse_options loc options;
  (* let path = Ppx_deriving.path_of_type_decl ~path type_decl in *)
  let type_string     = Ppx_deriving.mangle_type_decl (`Prefix "typestring") type_decl in
  let args            = Utils.get_params type_decl
                        |> List.map (fun a -> [%expr [%e a ] ()])
  in
  let typestring_func =
    Utils.application_str loc args (type_qualify loc path type_decl.ptype_name.txt)
  in
  let typestring_func = [%expr fun () -> [%e typestring_func]] in
  let typestring_type = typestring_type_of_decl loc ~options ~path type_decl in
  let typestring_var  = pvar type_string in
  [Vb.mk (Pat.constraint_ typestring_var typestring_type)
     (Ppx_deriving.poly_fun_of_type_decl type_decl typestring_func)]

let type_decl_str ~options ~path type_decls =
  [Str.value Nonrecursive
     (List.concat (List.map (str_of_type ~options ~path) type_decls));
  ]

let type_decl_sig ~options ~path type_decls =
  List.concat (List.map (sig_of_type ~options ~path) type_decls)

let deriver = Ppx_deriving.create deriver_name ~type_decl_str ~type_decl_sig ()
