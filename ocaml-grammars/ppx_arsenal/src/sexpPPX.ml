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

open Longident
open Asttypes
open Parsetree
open Ast_helper
open Ppx_deriving.Ast_convenience

open Utils
   
let deriver_name = "sexp"

(* Parse Tree and PPX Helpers *)

let argn = Printf.sprintf "a%d"

let pattn typs    = List.mapi (fun i _ -> pvar (argn i)) typs
let pattn_named l =
  let aux i x = mkloc (Lident x.pld_name.txt) x.pld_name.loc, pvar (argn i) in
  List.mapi aux l

let parse_options = List.iter @@ fun (name, pexp) ->
                                 match name with
                                 | _ ->
                                    raise_errorf ~loc:pexp.pexp_loc
                                      "The %s deriver takes no option %s." deriver_name name

(* Generator Type *)

let sexp_type_of_decl ~options ~path:_path type_decl =
  let loc = type_decl.ptype_loc in
  parse_options options;
  let typ = Ppx_deriving.core_type_of_type_decl type_decl in
  let typ =
    match type_decl.ptype_manifest with
    | Some {ptyp_desc = Ptyp_variant (_, Closed, _) ; _ } ->
       let row_field = {
           prf_desc = Rinherit typ;
           prf_loc = typ.ptyp_loc;
           prf_attributes = [];
         }
       in
       {ptyp_desc = Ptyp_variant ([row_field], Open, None);
        ptyp_loc  = Location.none;
        ptyp_loc_stack  = [];
        ptyp_attributes = []}
    | _ -> typ
  in
  Ppx_deriving.poly_arrow_of_type_decl
    (fun var -> [%type: ([%t var] -> Sexp.t) * string ])
    type_decl [%type: [%t typ] -> Sexp.t ]

(* Generator Function *)
let atom cnstr typestring_expr =
  let loc = typestring_expr.pexp_loc in
  [%expr PPX_Sexp.constructor [%e str2exp cnstr ] [%e typestring_expr ] ]

let expr_of_typ typestring_expr typ =
  let rec expr_of_typ x : expression =
    let loc = x.ptyp_loc in
    match x with

    (* Referencing another type, possibly polymorphic: typs are the types arguments *)
    | {ptyp_desc = Ptyp_constr ({txt = lid ; _}, typs) ; _} ->
       let sexp    = ident "sexp_of" lid in
       let typestr = ident "typestring" lid in
       let args    = List.map expr_of_typ typs in
       let typestr_args = List.map (esnd loc) args in
       [%expr [%e app sexp args], [%e app typestr typestr_args]]

    (* typ is a product type: we don't deal with those *)
    | {ptyp_desc = Ptyp_tuple _ ; _} ->
       raise_errorf "Please do not use tuples in deriver %s, always use a constructor." deriver_name

    (* typ is a variant type: we construct the expression pair (f,typestring),
         where f is a pattern-matching function over inhabitants of typ,
         and typestring is the string representing typ *)
    | {ptyp_desc = Ptyp_variant (fields, _, _) ; _} ->

       (* treat_field constructs a pattern-matching case for one constructor (field) *)
       let treat_field field =
         let variant label popt = Pat.variant label.txt popt in
         match field with
         | Rtag (label, true, []) ->
            Exp.case (variant label None) (atom label.txt typestring_expr)
         | Rtag (label, false, typs) ->
            let aux i typ = [%expr fst [%e expr_of_typ typ] [%e evar (argn i)] ] in
            let aux' sofar e = [%expr [%e e] :: [%e sofar] ] in
            let args = typs |> List.mapi aux |> List.fold_left aux' [%expr [] ] in
            Exp.case (variant label (Some [%pat? x]))
              [%expr Sexp.List ([%e atom label.Location.txt typestring_expr] :: List.rev [%e args] ) ] 

         | Rinherit({ ptyp_desc = Ptyp_constr (tname, _) ; _ } as typ) ->
            Exp.case [%pat? [%p Pat.type_ tname] as x] (expr_of_typ typ)
         | _ ->
            raise_errorf ~loc:typ.ptyp_loc "Cannot derive %s for %s."
              deriver_name (Ppx_deriving.string_of_core_type typ)
       in
       let treat_field field = treat_field field.prf_desc in
       let cases = fields |> List.map treat_field in
       [%expr [%e Exp.function_ cases], [%e typestring_expr ] ]

    (* typ is one of our type parameters: we have been given the value as an argument *)
    | {ptyp_desc = Ptyp_var name ; _} -> evar ("poly_" ^ name)
    (* typ is an alias: we traverse *)
    | {ptyp_desc = Ptyp_alias (typ, _) ; _} -> expr_of_typ typ
    (* Can't deal with any other kinds of types *)
    | {ptyp_loc ; _} ->
       raise_errorf ~loc:ptyp_loc "Cannot derive %s for %s."
         deriver_name (Ppx_deriving.string_of_core_type typ)
  in
  expr_of_typ typ    

let expr_of_type_decl ~path type_decl =
  let loc = type_decl.ptype_loc in
  let typestring_expr =
    let aux param sofar =
      let param =
        "poly_"^param.txt
        |> Lexing.from_string
        |> Parse.longident
        |> mknoloc
        |> Exp.ident
      in
      [%expr [%e sofar]^"("^(snd [%e param])^")"]
    in
    Ppx_deriving.fold_right_type_decl aux type_decl
      (fully_qualified path type_decl.ptype_name.txt |> str2exp)
  in
  match type_decl.ptype_kind, type_decl.ptype_manifest with
  | Ptype_abstract, Some manifest -> expr_of_typ typestring_expr manifest |> efst loc

  | Ptype_variant cs, _ ->

     let treat_field { pcd_name = { txt = name' ; _ }; pcd_args ; _ } =
       let aux pat typs =
         let aux i typ =
           [%expr fst [%e expr_of_typ typestring_expr typ] [%e evar (argn i)] ]
         in
         let aux' sofar e = [%expr [%e e] :: [%e sofar] ] in
         let args = typs |> List.mapi aux |> List.fold_left aux' [%expr [] ] in
         let qualifname' = fully_qualified path name' in
         Exp.case pat
           [%expr Sexp.List ( [%e atom qualifname' typestring_expr] :: List.rev [%e args] ) ]
       in
       match pcd_args with

       | Parsetree.Pcstr_tuple [] ->
          let qualifname' = fully_qualified path name' in
          Exp.case (pconstr name' []) (atom qualifname' typestring_expr)

       | Parsetree.Pcstr_tuple typs -> aux (pconstr name' (pattn typs)) typs

       | Parsetree.Pcstr_record l ->
          let typs = List.map (fun x -> x.pld_type) l in
          let pconstr name args =
            let args = Some (Ast_helper.Pat.record args Closed) in
            Ast_helper.Pat.construct (mknoloc (Lident name)) args
          in
          aux (pconstr name' (pattn_named l)) typs

     in
     
     let cases = cs |> List.map treat_field in
     Exp.function_ cases

  | Ptype_record _fields, _ ->
     raise_errorf ~loc "Cannot derive %s for record type." deriver_name
  | Ptype_abstract, None ->
     raise_errorf ~loc "Cannot derive %s for fully abstract type." deriver_name
  | Ptype_open, _ ->
     raise_errorf ~loc "Cannot derive %s for open type." deriver_name

    

(* Signature and Structure Components *)

let sig_of_type ~options ~path type_decl =
  parse_options options;
  [Sig.value
     (Val.mk
        (mknoloc (Ppx_deriving.mangle_type_decl (`Prefix "sexp_of") type_decl))
        (sexp_type_of_decl ~options ~path type_decl))]

let str_of_type ~options ~path type_decl =
  parse_options options;
  (* let path = Ppx_deriving.path_of_type_decl ~path type_decl in *)
  let sexp_func = expr_of_type_decl ~path type_decl in
  let sexp_type = sexp_type_of_decl ~options ~path type_decl in
  let sexp_var =
    pvar (Ppx_deriving.mangle_type_decl (`Prefix "sexp_of") type_decl) in
  [Vb.mk (Pat.constraint_ sexp_var sexp_type)
     (Ppx_deriving.poly_fun_of_type_decl type_decl sexp_func)]

let type_decl_str ~options ~path type_decls =
  [Str.value Recursive
     (List.concat (List.map (str_of_type ~options ~path) type_decls)) ]

let type_decl_sig ~options ~path type_decls =
  List.concat (List.map (sig_of_type ~options ~path) type_decls)

let deriver = Ppx_deriving.create deriver_name ~type_decl_str ~type_decl_sig ()
