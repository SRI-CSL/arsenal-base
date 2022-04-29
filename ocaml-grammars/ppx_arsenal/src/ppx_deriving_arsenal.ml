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

open Containers

let filter recognized_options (key, _) = List.mem ~eq:String.equal key recognized_options
let filter recognized_options options = List.filter (filter recognized_options) options

let ( #+ ) (deriver1, options1) (deriver2, options2) = 
  let ( @@ ) f1 f2 = fun ~options ~path l ->
    let options1 = filter options1 options in
    let options2 = filter options2 options in
    f1 ~options:options1 ~path l @ f2 ~options:options2 ~path l
  in
  let open Ppx_deriving in
  let type_decl_str = deriver1.type_decl_str @@ deriver2.type_decl_str in
  let type_ext_str = deriver1.type_ext_str @@ deriver2.type_ext_str in
  let module_type_decl_str =
    deriver1.module_type_decl_str @@ deriver2.module_type_decl_str
  in
  let type_decl_sig = deriver1.type_decl_sig @@ deriver2.type_decl_sig in
  let type_ext_sig = deriver1.type_ext_sig @@ deriver2.type_ext_sig in
  let module_type_decl_sig =
    deriver1.module_type_decl_sig @@ deriver2.module_type_decl_sig
  in
  Ppx_deriving.create ""
    ~type_decl_str
    ~type_ext_str
    ~module_type_decl_str
    ~type_decl_sig
    ~type_ext_sig
    ~module_type_decl_sig
    (),
  List.union ~eq:String.equal options1 options2

let ( #++ ) deriver1 deriver2 =
  match Ppx_deriving.lookup deriver2 with
  | None -> failwith("Deriver "^ deriver2 ^ " was not loaded; check the libraries in your build")
  | Some deriver2 -> deriver1 #+ (deriver2, [])

                   
let arsenal,_ = (TypeString.deriver, [])
                  #++ "ord"
                        #+ (Serialise.deriver, [])
                             #++ "random"
                                   #+ (JSONdesc.deriver, ["in_grammar"])

let () = Ppx_deriving.register Serialise.deriver;;
let () = Ppx_deriving.register TypeString.deriver;;
let () = Ppx_deriving.register JSONdesc.deriver;;
let () = Ppx_deriving.register { arsenal with name = "arsenal" };;
