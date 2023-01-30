(* Copyright (C) 2015--2016  Petter A. Urkedal <paurkedal@gmail.com>
 *
 * Edited by Stephane.Graham-Lengrand <disteph@gmail.com> (2019)
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

open Longident
open Asttypes
open Parsetree
open Ast_helper
open Ppx_deriving.Ast_convenience

let deriver = "random"

let set_level = ref None


(* Imports from Prime *)

let ( *< ) f g x = f (g x)

module Option = struct
  let map f = function None -> None | Some x -> Some (f x)
end

module List = struct
  include List

  let rec fold f xs accu =
    match xs with
    | [] -> accu
    | x :: xs' -> fold f xs' (f x accu)

  let fmap f xs =
    let rec loop ys = function
      | [] -> ys
      | x :: xs -> loop (match f x with None -> ys | Some y -> (y :: ys)) xs in
    rev (loop [] xs)
end


(* Parse Tree and PPX Helpers *)

let raise_errorf = Ppx_deriving.raise_errorf

let parse_options = List.iter @@ fun (name, pexp) ->
  match name with
  | "set_level" -> 
    begin
      match pexp with 
      | {pexp_desc = Pexp_constant (Pconst_integer(n,_)) ; _ } ->
        set_level := Some (int_of_string n)
        | _ -> raise_errorf ~loc:pexp.pexp_loc "unsupported value for set_level"
    end
  | _ ->
    raise_errorf ~loc:pexp.pexp_loc
                 "The %s deriver takes no option %s." deriver name

let tuple_opt = function
  | [] -> None
  | [arg] -> Some arg
  | args -> Some (Exp.tuple args)

let mapper = new Ppx_deriving.mapper

let mapped_expr e =
  Ok (mapper#expression e)

let get_random_fun attrs =
  attrs |> Ppx_deriving.attr ~deriver "random"
        |> Ppx_deriving.Arg.(get_attr ~deriver mapped_expr)

let get_custom_fun attrs =
  attrs |> Ppx_deriving.attr ~deriver "custom"
  |> Ppx_deriving.Arg.(get_attr ~deriver mapped_expr)

let float2exp f = string_of_float f |> Const.float |> Exp.constant
  
(* Get weights from attributes *)
let get_weight attrs =
  let level_conv x = match x with
    | {pexp_desc = Pexp_constant (Pconst_integer(n,_)) ; _ } ->
      Ok (`Int(x,int_of_string n))
    | _ -> Ok (`Fun x)
  in

  let curr_level = attrs 
  |> Ppx_deriving.attr ~deriver "level"
  |> Ppx_deriving.Arg.(get_attr ~deriver level_conv)
  in 
  
  let active = match !set_level  with 
  | None -> true
  | Some set_l -> 
    match curr_level with 
    | Some `Int(_, curr_l) when curr_l != set_l -> false
    | _ -> true

  in   let conv x = match active with 
    | false -> Ok (`Float(x,0.0))
    | true -> match x with
      | {pexp_desc = Pexp_constant (Pconst_float(n,_)) ; _ } -> Ok (`Float(x,float_of_string n))
      | {pexp_desc = Pexp_constant (Pconst_integer(n,_)) ; _ } ->
        let f = float_of_int(int_of_string n) in
        Ok (`Float(float2exp f,f))
      | _ -> Ok (`Fun x)
  in

  attrs |> Ppx_deriving.attr ~deriver "weight"
        |> Ppx_deriving.Arg.(get_attr ~deriver conv )

(* Check whether the weight is static (fixed int or float, not a function) *)

let weight_is_static attrs =
  match get_weight attrs with
  | None
  | Some (`Float _) -> true
  | _ -> false

(* Checks whether weights are static and uniform *)
    
let weights_are_uniform attrs =
  let rec aux ?weight = function
    | [] -> true
    | head::tail -> 
      match get_weight head, weight with
      | None, None               -> aux ~weight:1. tail
      | Some (`Float(_,n)), None -> aux ~weight:n tail
      | None, Some w when w = 1. -> aux ?weight tail
      | Some (`Float(_,n)), Some w when n = w -> aux ?weight tail
      | _ -> false
  in
  aux attrs

type weight = Static of float | Dynamic of Ppxlib.Parsetree.expression
  
(* Assumes the weight is static; gets it as a float. *)

let get_weight_float_static attrs =
  match get_weight attrs with
  | None -> 1.
  | Some (`Float(_,n)) -> n
  | Some (`Fun _)      -> assert false

(* General case where the weight is dynamic, i.e. depends on local state; gets the weight as a float when given the state. *)

let get_weight_float loc attrs rng =
  match get_weight attrs with
  | None               -> Static 1.
  | Some (`Float(_,f)) -> Static f
  | Some (`Fun f)      -> Dynamic [%expr ([%e f]) ([%e rng])]

let pcd_attributes pcd = pcd.pcd_attributes
                       
let rowfield_attributes { prf_attributes ; _ } = prf_attributes

            
(* Generator Type *)

let random_type_of_decl ~options ~path:_path type_decl =
  let loc = type_decl.ptype_loc in
  parse_options options;
  let typ = Ppx_deriving.core_type_of_type_decl type_decl in
  let typ =
    match type_decl.ptype_manifest with
    | Some {ptyp_desc = Ptyp_variant (_, Closed, _) ; _ } ->
       let row_field = {
           prf_desc = Rinherit typ;
           prf_loc  = typ.Parsetree.ptyp_loc ;
           prf_attributes = [];
         } in
       let ptyp_desc = Ptyp_variant ([row_field], Open, None) in
       {ptyp_desc; ptyp_loc = Location.none; ptyp_attributes = []; ptyp_loc_stack = []}
    | _ -> typ in
  Ppx_deriving.poly_arrow_of_type_decl
    (fun var -> [%type: PPX_Random.state -> [%t var]])
    type_decl
    [%type: PPX_Random.state -> [%t typ]]


(* Generator Function *)

(* Generates the list of cases when the weights are static *)

let cumulative_static get_attrs cs =
  let cs = List.map (fun pcd -> get_weight_float_static (get_attrs pcd), pcd) cs in
  let c_norm = 1.0 /. List.fold (fun (w, _) -> (+.) w) cs 0.0 in
  let normalize w =
    let x = int_of_float (ldexp (w *. c_norm) 30) in
    (* On 32 bit platforms, 1.0 ± ε will be mapped to min_int. *)
    if x < 0 then max_int else x in
  let cs = cs |> List.map (fun (w, pcd) -> (normalize w, pcd))
              |> List.sort (fun x y -> Int.compare (fst y) (fst x)) in
  fst @@ List.fold (fun (w, pcd) (acc, rem) -> (rem - w, pcd) :: acc, rem - w)
                   cs ([], 1 lsl 30)

(* Generates the list of cases when the weights are dynamic *)

let cumulative loc get_attrs cs rng =
  let cs = List.map (fun pcd -> get_weight_float loc (get_attrs pcd) rng, pcd) cs in
  let aux (w,_) (static, dynamic) =
    match w with
    | Static f  -> f +. static, dynamic
    | Dynamic e -> static, [%expr [%e e] +. [%e dynamic]]
  in
  let total_exp_static, total_exp_dynamic = List.fold aux cs (0. , [%expr 0.0] ) in
  let total_exp = [%expr [%e float2exp total_exp_static] +. [%e total_exp_dynamic]] in
  let prelude e =
    [%expr 
      let c_norm = [%e total_exp] in
      let normalize w =
        let x = int_of_float (ldexp (w /. c_norm) 30) in
        (* On 32 bit platforms, 1.0 ± ε will be mapped to min_int. *)
        if x < 0 then max_int else x in
      [%e e]
    ]
  in
  let close static dynamic =
    [%expr (1 lsl 30) - normalize([%e float2exp static] +. [%e dynamic]) ]
  in
  let aux (w,pcd) (acc, static, dynamic) =
    let static, dynamic =
      match w with
      | Static f  -> f +. static, dynamic
      | Dynamic e -> static, [%expr [%e e] +. [%e dynamic]]
    in
    (* let e = [%expr [%e rem] - normalize ()] in *)
    (close static dynamic, pcd) :: acc, static, dynamic
  in
  let acc, _, _ = List.fold aux cs ([], 0. , [%expr 0. ]) in
  prelude, acc

let invalid_case loc =
  Exp.case [%pat? i]
    [%expr
      failwith (Printf.sprintf "Value %d from PPX_Random.case is out of range." i)]

let rec expr_of_typ typ =
  let expr_of_rowfield = function
    | Rtag (label, true, []) -> Exp.variant label.Location.txt None
    | Rtag (label, false, typs) ->
      Exp.variant label.Location.txt (tuple_opt (List.map expr_of_typ typs))
    | Rinherit typ -> expr_of_typ typ
    | _ ->
      raise_errorf ~loc:typ.ptyp_loc "Cannot derive %s for %s."
        deriver (Ppx_deriving.string_of_core_type typ)
  in
  let expr_of_rowfield field =
    let loc = field.prf_loc in
    [%expr (fun rng -> [%e expr_of_rowfield field.prf_desc]) (PPX_Random.deepen rng)]
  in
  match get_random_fun typ.ptyp_attributes with
  | Some f ->
     let loc = f.pexp_loc in
     app f [[%expr rng]]
  | None ->
     let loc = typ.ptyp_loc in 
     match typ with
     | [%type: unit] -> [%expr ()]
     | {ptyp_desc = Ptyp_constr ({txt = lid ; _ }, typs); _} ->
        let f    =
          match get_custom_fun typ.ptyp_attributes with
          | Some f -> f
          | None -> Exp.ident (mknoloc (Ppx_deriving.mangle_lid (`Prefix "random") lid))
        in
        let args = List.map (fun typ -> [%expr fun rng -> [%e expr_of_typ typ]]) typs in
        app f (args @ [[%expr rng]])
     | {ptyp_desc = Ptyp_tuple typs ; _ } -> Exp.tuple (List.map expr_of_typ typs)

    (* The following case is an optimisation of the case below: if all weights are static and equal to 1., no need to run fancy code *)
     | {ptyp_desc = Ptyp_variant (fields, _, _); _ }
          when fields |> List.map rowfield_attributes |> weights_are_uniform  ->
        let make_case j field =
          Exp.case (j |> Const.int |> Pat.constant) (expr_of_rowfield field) in
        let cases = List.mapi make_case fields in
        let case_count = cases |> List.length |> Const.int |> Exp.constant in
        Exp.match_ [%expr PPX_Random.case [%e case_count ] rng] (cases @ [invalid_case loc])

    (* In the following case, weights are not all equal to 1. but are static, so we can compute probabilities statically *)
    | {ptyp_desc = Ptyp_variant (fields, _, _) ; _ }
      when List.for_all (weight_is_static *< rowfield_attributes) fields ->
      let branch (i, field) cont =
        [%expr if w > [%e Exp.constant (Const.int i)]
          then [%e expr_of_rowfield field]
          else [%e cont] ] in
      begin match cumulative_static rowfield_attributes fields with
        | [] -> assert false
        | (_, field) :: fields ->
          [%expr let w = PPX_Random.case_30b rng in
            [%e List.fold branch fields (expr_of_rowfield field)]]
      end

    (* In the general case, weights are dynamic so we compute probabilities at every call *)
    | {ptyp_desc = Ptyp_variant (fields, _, _); _ } ->
      let branch (iexp, field) cont =
        [%expr if w > [%e iexp]
               then [%e expr_of_rowfield field]
               else [%e cont] ] in
      let prelude,l = cumulative loc rowfield_attributes fields [%expr rng] in
      begin
        match l with
        | [] -> assert false (* There's at least one constructor *)
        | (_, field) :: fields ->
          prelude (
            [%expr let w = PPX_Random.case_30b rng in
              [%e List.fold branch fields (expr_of_rowfield field)]])
      end
    | {ptyp_desc = Ptyp_var name ; _ } -> [%expr [%e evar ("poly_" ^ name)] rng]
    | {ptyp_desc = Ptyp_alias (typ, _) ; _ } -> expr_of_typ typ
    | {ptyp_loc ; _ } ->
      raise_errorf ~loc:ptyp_loc "Cannot derive %s for %s."
                   deriver (Ppx_deriving.string_of_core_type typ)

let expr_of_type_decl ({ptype_loc = loc; _ } as type_decl) =
  let expr_of_constr pcd =
    let lid = {txt = Lident pcd.pcd_name.txt; loc = pcd.pcd_name.loc} in
    match pcd.pcd_args with
    | Parsetree.Pcstr_tuple l ->
      Exp.construct lid (tuple_opt (List.map expr_of_typ l))
    | Parsetree.Pcstr_record l ->
       let aux (label_dec : label_declaration) =
         let lid = {txt = Lident label_dec.pld_name.txt; loc = label_dec.pld_name.loc} in
         (lid, expr_of_typ label_dec.pld_type)
       in
       let args = List.map aux l in
       Exp.construct lid (Some(Exp.record args None))
  in
  let expr_of_constr field = [%expr (fun rng -> [%e expr_of_constr field]) (PPX_Random.deepen rng)] in
  match type_decl.ptype_kind, type_decl.ptype_manifest with
  | Ptype_abstract, Some manifest ->
    [%expr fun rng -> [%e expr_of_typ manifest]]
  | Ptype_variant constrs, _
    when constrs |> List.map pcd_attributes |> weights_are_uniform  ->
    let make_case j pcd = Exp.case (j |> Const.int |> Pat.constant) (expr_of_constr pcd) in
    let cases = List.mapi make_case constrs in
    let case_count = cases |> List.length |> Const.int |> Exp.constant in
    [%expr fun rng ->
      [%e Exp.match_ [%expr PPX_Random.case [%e case_count] rng] (cases @ [invalid_case loc])] ]

  | Ptype_variant cs, _
    when List.for_all (weight_is_static *< pcd_attributes) cs ->
    let branch (w, pcd) cont =
      [%expr if w > [%e Exp.constant (Const.int w)]
        then [%e expr_of_constr pcd]
        else [%e cont] ] in
    begin match cumulative_static pcd_attributes cs with
      | [] -> assert false
      | (_, pcd) :: cs ->
        [%expr fun rng -> let w = PPX_Random.case_30b rng in
          [%e List.fold branch cs (expr_of_constr pcd)] ]
    end

  | Ptype_variant cs, _ ->
    let branch (iexp, pcd) cont =
      [%expr if w > [%e iexp]
             then [%e expr_of_constr pcd]
             else [%e cont] ] in
    let prelude,l = cumulative loc pcd_attributes cs [%expr rng] in
    begin match l with
    | [] -> assert false (* There's at least one constructor *)
    | (_, pcd) :: cs ->
      [%expr fun rng ->
        [%e prelude
            [%expr let w = PPX_Random.case_30b rng in
              [%e List.fold branch cs (expr_of_constr pcd)] ]]]
    end
  | Ptype_record fields, _ ->
    let fields = fields |> List.map @@ fun pld ->
      {txt = Lident pld.pld_name.txt; loc = pld.pld_name.loc},
      expr_of_typ pld.pld_type in
    [%expr fun rng -> [%e Exp.record fields None]]
  | Ptype_abstract, None ->
    raise_errorf ~loc "Cannot derive %s for fully abstract type." deriver
  | Ptype_open, _ ->
    raise_errorf ~loc "Cannot derive %s for open type." deriver


(* Signature and Structure Components *)

let sig_of_type ~options ~path type_decl =
  parse_options options;
  [Sig.value
    (Val.mk
      (mknoloc (Ppx_deriving.mangle_type_decl (`Prefix "random") type_decl))
      (random_type_of_decl ~options ~path type_decl))]

let str_of_type ~options ~path type_decl =
  parse_options options;
  let path = Ppx_deriving.path_of_type_decl ~path type_decl in
  let random_func = expr_of_type_decl type_decl in
  let random_type = random_type_of_decl ~options ~path type_decl in
  let random_var =
    pvar (Ppx_deriving.mangle_type_decl (`Prefix "random") type_decl) in
  [Vb.mk (Pat.constraint_ random_var random_type)
         (Ppx_deriving.poly_fun_of_type_decl type_decl random_func)]

let () =
  Ppx_deriving.register @@
  Ppx_deriving.create deriver
    ~core_type: (fun typ -> let loc = typ.ptyp_loc in [%expr fun rng -> [%e expr_of_typ typ]])
    ~type_decl_str: (fun ~options ~path type_decls ->
      [Str.value Recursive
        (List.concat (List.map (str_of_type ~options ~path) type_decls))])
    ~type_decl_sig: (fun ~options ~path type_decls ->
      List.concat (List.map (sig_of_type ~options ~path) type_decls))
    ()
