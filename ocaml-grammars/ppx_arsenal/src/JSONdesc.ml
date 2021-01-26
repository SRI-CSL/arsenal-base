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
   
let deriver_name = "json_desc"

let parse_options l =
  let result = ref true in
  let aux (name, pexp) = 
    match name with
    | "in_grammar" -> result := Ppx_deriving.Arg.(get_expr ~deriver:deriver_name bool) pexp
    | _ ->
       raise_errorf ~loc:pexp.pexp_loc
         "The %s deriver takes no option %s." deriver_name name
  in
  List.iter aux l;
  !result

let json_desc_type_of_decl ~options:_ ~path:_path type_decl =
  let loc = type_decl.ptype_loc in
  Ppx_deriving.poly_arrow_of_type_decl
    (fun _var -> [%type: string])
    type_decl
    [%type: unit -> unit]

let rec call loc lid typs =
  let args    = List.map (fun x -> let y,_,_ = expr_of_typ x in y) typs in
  let record_json = ident "json_desc" lid in
  let typestr = ident "typestring" lid in
  [%expr [%e app record_json args] (); [%e app typestr args] ]

and expr_of_typ typ : expression*bool*bool =
  match typ with

  (* Referencing an option type: typs are the types arguments *)
  | {ptyp_desc = Ptyp_constr ({txt = Lident "option" ; loc }, [arg]) ; _} ->
     let args,optional,list = expr_of_typ arg in
     if optional
     then raise_errorf ~loc "Deriver %s doe snot accept option of option." deriver_name;
     args, true, list

  (* Referencing an option type: typs are the types arguments *)
  | {ptyp_desc = Ptyp_constr ({txt = Lident "list" ; loc }, [arg]) ; _} ->
     let args,optional,list = expr_of_typ arg in
     if optional || list
     then raise_errorf ~loc "Deriver %s doe snot accept list of option." deriver_name;
     args, false, true

  (* Referencing another type, possibly polymorphic: typs are the types arguments *)
  | {ptyp_desc = Ptyp_constr ({txt = lid ; loc }, typs) ; _} ->
     call loc lid typs, false, false

  (* typ is a product type: we don't deal with those *)
  | {ptyp_desc = Ptyp_tuple _ ; _} ->
     raise_errorf "Please do not use tuples in deriver %s, always use a constructor." deriver_name

  (* typ is a variant type: we construct the expression pair (f,typestring),
         where f is a pattern-matching function over inhabitants of typ,
         and typestring is the string representing typ *)
  | {ptyp_desc = Ptyp_variant ([field], _, _) ; _} ->
     (* treat_field constructs a pattern-matching case for one constructor (field) *)
     begin
       match field.prf_desc with
       | Rtag (label, _, []) -> str2exp label.txt, false, false

       | _ ->
          raise_errorf ~loc:typ.ptyp_loc "Cannot derive %s for %s."
            deriver_name (Ppx_deriving.string_of_core_type typ)
     end
  (* typ is one of our type parameters: we have been given the value as an argument *)
  | {ptyp_desc = Ptyp_var name ; _} -> evar ("poly_" ^ name), false, false
  (* typ is an alias: we traverse *)
  | {ptyp_desc = Ptyp_alias (typ, _) ; _} -> expr_of_typ typ
  (* Can't deal with any other kinds of types *)
  | {ptyp_loc ; _} ->
     raise_errorf ~loc:ptyp_loc "Cannot derive %s for %s."
       deriver_name (Ppx_deriving.string_of_core_type typ)

let argn i = Printf.sprintf "arg%d" i

type argument = {
    name : string;
    typ  : expression;
    optional : bool;
    list : bool
  }

let prefix loc s = [%expr `String ("#/definitions/"^[%e s ]) ]

let build_alternative loc cons args : expression * expression =
  let cons       = str2exp cons in
  let req arg    = not arg.optional in
  let name arg   = str2exp arg.name in
  let nameString arg = [%expr `String [%e name arg]] in
  let required   = List.filter req args |> List.map nameString |> list loc in
  let format arg =
    let common = [%expr `Assoc ["$ref", [%e prefix loc arg.typ ] ] ] in
    if arg.list then
      [%expr [%e name arg],
       `Assoc ["type", `String "array" ;
               "items", [%e common ] ] ]
    else
      [%expr [%e name arg], [%e common] ]
  in
  prefix loc cons,
  [%expr
      [%e cons],
   `Assoc [
       "type", `String "object";
       "required", `List (`String "constructor"::[%e required]);
       "properties",
       `Assoc 
         (("constructor", `Assoc [ "type",    `String "string";
                                   "pattern", `String [%e cons] ])
          :: [%e List.map format args |> list loc ])  ] ]

(* let build_alternative loc cons args : expression =
 *   [%expr `Assoc [ "constructor", `String [%e str2exp cons] ;
 *                   "arguments", `List [%e list loc args] ] ] *)


let build_type loc typ alternatives : expression =
  let aux (name,_) = [%expr `Assoc [ "$ref", [%e name]]] in
  let references = List.map aux alternatives |> list loc in
  let init = 
    [%expr
        [ [%e typ],
          `Assoc ["anyOf", `List [%e references ] ] ] ]
  in
  List.map snd alternatives |> list loc ~init
  
let build_abbrev loc typ body : expression =
  [%expr
      [ [%e typ],
        `Assoc ["anyOf", `List [ `Assoc [ "$ref", [%e body]] ] ] ] ]
  
(* Treating a type declaration 
       type foo = ...
     possibly
       type ('a,...,'n) foo = ...

     Producing an expression, of type string, describing the type declaration
 *)
  
let expr_of_type_decl ~path typestring_expr type_decl =
  let loc = type_decl.ptype_loc in (* location of the type declaration *)
  match type_decl.ptype_kind, type_decl.ptype_manifest with
  | Ptype_abstract, Some {ptyp_desc = Ptyp_constr ({txt = lid ; _}, typs) ; _} ->
     build_abbrev loc typestring_expr (prefix loc (call loc lid typs))

  | Ptype_variant cs, _ -> (* foo is a variant type with a series of constructors C1 ... Cm *)

     let treat_field { pcd_name = { txt = name' ; _ }; pcd_args ; _ } =
       (* This treats one particular construction C of t1 * ... * tp
            name' is C *)
       match pcd_args with

       | Parsetree.Pcstr_tuple typs -> (* typs is the list t1 ... tp *)
          (* we build the JSON { "constructor" : "C"; "arguments" : args } *)
          let aux i typ =
            let typ, optional, list = expr_of_typ typ in
            { name = argn i; typ; optional; list }
          in
          let args = typs |> List.mapi aux in
          build_alternative loc (fully_qualified path name') args

       | Parsetree.Pcstr_record args ->
          let aux (x : label_declaration) =
            let typ, optional, list = expr_of_typ x.pld_type in
            { name = x.pld_name.txt; typ; optional; list }
          in
          let args = args |> List.map aux in
          build_alternative loc (fully_qualified path name') args
     in
     
     cs |> List.map treat_field |> build_type loc typestring_expr

  | Ptype_record _fields, _ ->
     raise_errorf ~loc "Cannot derive %s for record type." deriver_name
  | Ptype_abstract, Some _ ->
     raise_errorf ~loc "Cannot derive %s for abbreviation with that kind of body." deriver_name
  | Ptype_abstract, None ->
     raise_errorf ~loc "Cannot derive %s for fully abstract type." deriver_name
  | Ptype_open, _ ->
     raise_errorf ~loc "Cannot derive %s for open type." deriver_name

let expr_of_type_decl in_grammar ~path type_decl =
  let loc = type_decl.ptype_loc in (* location of the type declaration *)
  let typestring_expr = (* building the string "foo(poly_a)...(poly_n)"*)
    let aux param sofar =
      ("poly_"^param.txt
       |> Lexing.from_string
       |> Parse.longident
       |> mknoloc
       |> Exp.ident)::sofar
    in
    let l = Ppx_deriving.fold_right_type_decl aux type_decl [] in
    let name = ident "typestring" (Lident type_decl.ptype_name.txt) in
    app name l
  in
  if in_grammar then
    [%expr
        fun () ->
        if not(JSONindex.mem [%e typestring_expr])
        then
          begin
            let mark = JSONindex.mark [%e typestring_expr] in
            let json_list = [%e expr_of_type_decl ~path typestring_expr type_decl ] in
            Format.(fprintf err_formatter) "Registering type %s\n" [%e typestring_expr];
            JSONindex.add mark json_list
          end
    ]
  else
    [%expr fun () ->
        Format.(fprintf err_formatter) "Skipping type %s\n" [%e typestring_expr]]

(* Signature and Structure Components *)

let sig_of_type ~options ~path type_decl =
  [Sig.value
     (Val.mk
        (mknoloc (Ppx_deriving.mangle_type_decl (`Prefix "json_desc") type_decl))
        (json_desc_type_of_decl ~options ~path type_decl))]

let str_of_type ~options ~path type_decl =
  let in_grammar = parse_options options in
  let sexp_func = expr_of_type_decl in_grammar ~path type_decl in
  let path = Ppx_deriving.path_of_type_decl ~path type_decl in
  let sexp_type = json_desc_type_of_decl ~options ~path type_decl in
  let sexp_var =
    pvar (Ppx_deriving.mangle_type_decl (`Prefix "json_desc") type_decl) in
  [Vb.mk (Pat.constraint_ sexp_var sexp_type)
     (Ppx_deriving.poly_fun_of_type_decl type_decl sexp_func)]

let type_decl_str ~options ~path type_decls =
  [Str.value Recursive
     (List.concat (List.map (str_of_type ~options ~path) type_decls)) ]

let type_decl_sig ~options ~path type_decls =
  List.concat (List.map (sig_of_type ~options ~path) type_decls)

let deriver = Ppx_deriving.create deriver_name ~type_decl_str ~type_decl_sig ()
