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
   
let deriver_name = "json_desc"

let in_grammar = ref true

let parse_options loc l =
  in_grammar := true; (* Unless otherwise stated, type declaration is in grammar *)
  with_path := None;  (* Use runtime option for determining how to qualify paths *)
  let aux (name, pexp) = 
    match name with
    | "in_grammar" -> in_grammar := Ppx_deriving.Arg.(get_expr ~deriver:deriver_name bool) pexp
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

let json_desc_type_of_decl ~options:_ ~path:_path type_decl =
  let loc = type_decl.ptype_loc in
  Ppx_deriving.poly_arrow_of_type_decl
    (fun _var -> [%type: unit -> string])
    type_decl
    [%type: unit -> unit]

let key_type_of_decl ~options ~path:_path type_decl =
  let loc = type_decl.ptype_loc in
  parse_options loc options;
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
    (fun _var -> [%type: unit ])
    type_decl [%type: [%t typ] TUID.t ]

let rec call loc lid typs =
  let args        = List.map (fun x -> let y,_,_ = expr_of_typ x in y) typs in
  let record_json = ident "json_desc" lid in
  let typestr     = ident "typestring" lid in
  [%expr [%e app record_json args] (); [%e app typestr args] ]

and expr_of_typ typ : expression*bool*bool =
  match typ with

  (* Referencing an option type: typs are the types arguments *)
  | {ptyp_desc = Ptyp_constr ({txt = Lident "option" ; loc }, [arg]) ; _} ->
     let args,optional,list = expr_of_typ arg in
     if optional
     then raise_errorf ~loc "Deriver %s does not accept option of option." deriver_name;
     args, true, list

  (* Referencing a list type: typs are the types arguments *)
  | {ptyp_desc = Ptyp_constr ({txt = Lident "list" ; loc }, [arg]) ; _} ->
     let args,optional,list = expr_of_typ arg in
     if optional || list
     then raise_errorf ~loc "Deriver %s does not accept list of option." deriver_name;
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

let prefix_noapp loc s = [%expr `String ("#/definitions/"^[%e s ]) ]
let prefix loc s = [%expr `String ("#/definitions/"^[%e s ] ()) ]

let build_alternative loc qcons cons args : expression * expression =
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
  prefix_noapp loc qcons,
  [%expr
      [%e qcons],
   `Assoc [
       "type", `String "object";
       "additionalProperties", `Bool false;
       "required", `List (`String PPX_Serialise.json_constructor_field::[%e required]);
       "properties",
       `Assoc 
         ((PPX_Serialise.json_constructor_field, `Assoc [ "type",    `String "string";
                                    "pattern", `String [%e cons] ])
          :: [%e List.map format args |> list loc ])  ] ]

(* let build_alternative loc cons args : expression =
 *   [%expr `Assoc [ PPX_Serialise.json_constructor_field, `String [%e str2exp cons] ;
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
  
let expr_of_type_decl ~path poly_args typestring_expr type_decl =
  let loc = type_decl.ptype_loc in (* location of the type declaration *)
  match type_decl.ptype_kind, type_decl.ptype_manifest with
  | Ptype_abstract, Some {ptyp_desc = Ptyp_constr ({txt = lid ; _}, typs) ; _} ->
     build_abbrev loc typestring_expr (prefix loc (call loc lid typs))

  | Ptype_variant cs, _ -> (* foo is a variant type with a series of constructors C1 ... Cm *)

     let treat_field { pcd_name = { txt = name' ; _ }; pcd_args ; _ } =
       (* This treats one particular construction C of t1 * ... * tp
            name' is C *)
       (* First, we qualify the constructor's name with the type parameters *)
       let qname = application_str loc poly_args (constructor_qualify loc path name') in
       match pcd_args with

       | Parsetree.Pcstr_tuple typs -> (* typs is the list t1 ... tp *)
          (* we build the JSON { PPX_Serialise.json_constructor_field : "C"; "arguments" : args } *)
          let aux i typ =
            let typ, optional, list = expr_of_typ typ in
            { name = argn i; typ; optional; list }
          in
          let args = typs |> List.mapi aux in
          build_alternative loc qname (str2exp name') args

       | Parsetree.Pcstr_record args ->
          let aux (x : label_declaration) =
            let typ, optional, list = expr_of_typ x.pld_type in
            { name = x.pld_name.txt; typ; optional; list }
          in
          let args = args |> List.map aux in
          build_alternative loc qname (str2exp name') args
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

let expr_of_type_decl in_grammar ~path ~about type_decl : expression * expression * expression=
  let loc = type_decl.ptype_loc in (* location of the type declaration *)
  (* We first build the string "foo(poly_a)...(poly_n)"*)
  let args = Utils.get_params type_decl in
  let args_applied = args |> List.map (fun a -> [%expr [%e a] ()]) in
  let typestring_expr =
    let name = ident "typestring" (Lident type_decl.ptype_name.txt) in
    app (app name args) [ [%expr ()] ]
  in
  let key =
    match args with
    | [] ->
       let id      = Lident type_decl.ptype_name.txt in
       let random  = ident "random"    id in
       let name    = ident "serialise" id in
       let hash    = [%expr [%e name].PPX_Serialise.hash] in
       let compare = [%expr [%e name].PPX_Serialise.compare] in
       let name    = [%expr [%e name].PPX_Serialise.typestring()] in
       [%expr TUID.create
           ~hash:[%e hash]
           ~compare:[%e compare]
           ~random:[%e random]
           ~name:[%e name] ]
    | _::_ -> [%expr failwith "Can't get key for polymorphic type"]
  in
  if in_grammar then
    [%expr
     fun () ->
       if not(JSONindex.mem [%e typestring_expr])
       then
         begin
           let mark = JSONindex.mark [%e typestring_expr] in
           let json_list = [%e expr_of_type_decl ~path args_applied typestring_expr type_decl ] in
           Format.(fprintf err_formatter) "Registering type %s\n" [%e typestring_expr];
           JSONindex.add mark json_list
         end
    ],
    begin
      match args with
      | [] ->
         [%expr
             if not(Register.mem [%e typestring_expr])
             then
               Register.add [%e typestring_expr] [%e about];
         ]
      | _::_ -> [%expr ()]
    end,
    key
  else
    [%expr fun () ->
        Format.(fprintf err_formatter) "Skipping type %s\n" [%e typestring_expr]],
    [%expr ()],
    key
    

(* Signature and Structure Components *)

let sig_of_type ~options ~path type_decl =
  [Sig.value
     (Val.mk
        (mknoloc (Ppx_deriving.mangle_type_decl (`Prefix "json_desc") type_decl))
        (json_desc_type_of_decl ~options ~path type_decl));
   Sig.value
     (Val.mk
        (mknoloc (Ppx_deriving.mangle_type_decl (`Prefix "key") type_decl))
        (key_type_of_decl ~options ~path type_decl))
  ]

let str_of_type ~options ~path type_decl =
  let loc          = type_decl.ptype_loc in (* location of the type declaration *)
  parse_options loc options;
  let key          = Ppx_deriving.mangle_type_decl (`Prefix "key") type_decl in
  let json_desc    = Ppx_deriving.mangle_type_decl (`Prefix "json_desc") type_decl in
  let serialise    = ident_decl "serialise" type_decl in
  let about =
    [%expr
        About.About{
          key       = [%e evar key ];
          json_desc = [%e evar json_desc ];
          serialise = [%e serialise ];
        }
    ]
  in
  let pvarname     = pvar json_desc in
  let pvarname3    = pvar key in
  let path         = Ppx_deriving.path_of_type_decl ~path type_decl in
  let func, record, key = expr_of_type_decl !in_grammar ~path ~about type_decl in
  let typ          = json_desc_type_of_decl ~options ~path type_decl in
  let loc          = type_decl.ptype_loc in
  let typ2         = [%type: unit] in
  let typ3         = key_type_of_decl ~options ~path type_decl in
  [Vb.mk (Pat.constraint_ pvarname typ) (Ppx_deriving.poly_fun_of_type_decl type_decl func)],
  [Vb.mk (Pat.constraint_ pvarname3 typ3) (Ppx_deriving.poly_fun_of_type_decl type_decl key)],
  [Vb.mk (Pat.constraint_ (punit()) typ2) record]

let type_decl_str ~options ~path type_decls =
  let l = List.map (str_of_type ~options ~path) type_decls in
  let a = List.map (fun (x,_,_) -> x) l in
  let b = List.map (fun (_,y,_) -> y) l in
  let c = List.map (fun (_,_,z) -> z) l in
  [Str.value Recursive (List.concat a);
   Str.value Nonrecursive (List.concat b);
   Str.value Nonrecursive (List.concat c)]

let type_decl_sig ~options ~path type_decls =
  List.concat (List.map (sig_of_type ~options ~path) type_decls)

let deriver = Ppx_deriving.create deriver_name ~type_decl_str ~type_decl_sig ()
