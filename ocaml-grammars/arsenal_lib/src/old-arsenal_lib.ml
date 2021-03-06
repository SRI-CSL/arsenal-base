open Format

open Sexplib
open Std

module JSON = Yojson.Safe

(************************************************************)
(* Conversions between json, S-expressions, Polish notation *)

exception Conversion of string

module PPX_Sexp = struct  
  let print_types = ref false (* Does not print types in S-expressions *)

  let atom = ":"
  let constructor cst ty =
    if !print_types then Sexp.List [Sexp.Atom ":"; Sexp.Atom cst; Sexp.Atom ty]
    else Sexp.Atom cst

  let is_atom = function
    | Sexp.List [Sexp.Atom ":"; Sexp.Atom _; Sexp.Atom _] when !print_types -> true
    | Sexp.Atom _ when not !print_types -> true
    | _ -> false
  
  let throw ?(who="PPX_Sexp") sexp =
    raise(Conversion(who^": "^Sexp.to_string sexp^" is not a good S-expression"))

  let get_cst ?who = function
    | Sexp.List [Sexp.Atom ":"; Sexp.Atom c; Sexp.Atom _] when !print_types -> c
    | Sexp.Atom c when not !print_types -> c
    | sexp -> throw ?who sexp

  let rec get_type ?who = function
    | Sexp.List [Sexp.Atom ":"; Sexp.Atom _; Sexp.Atom ty] when !print_types -> ty
    | Sexp.List (s::_)  when !print_types && is_atom s -> get_type ?who s
    | sexp -> throw ?who sexp

end


let exn f a = match f a with
  | Ok a -> a
  | Error s -> raise (Conversion s)

let rec json_of_sexp sexp =
  match sexp with
  | Sexp.Atom "true"  -> `Bool true
  | Sexp.Atom "false" -> `Bool false
  | Sexp.Atom "None"    -> `Null
  | Sexp.List[Sexp.Atom "Some"; a] -> json_of_sexp a
  | Sexp.Atom "Nil"    -> `List[]
  | Sexp.List(Sexp.Atom "List" :: l) -> `List(List.map json_of_sexp l)
  | Sexp.List(Sexp.Atom s::t) -> `List(`String s::List.map json_of_sexp t)
  | Sexp.Atom s     -> `List[`String s]
  | Sexp.List l     -> `List(List.map json_of_sexp l)

let rec json_clean json = match json with
  | `Assoc _  -> JSON.Util.member "node" json |> json_clean
  | `List l   -> `List(List.map json_clean l)
  | `String _ | `Null | `Bool _ | `Int _ -> json
  | _ -> raise (Conversion("clean_types: this JSON is not good: "^JSON.to_string json))
                     
let rec json_lift json = match json with
  | `List[`Null as i]
    | `List[`Bool _ as i]
    | `List[`Int _ as i] -> i
  | `List((`String _ as node)::l) -> `List(node::List.map json_lift l)
  | `List l    -> `List(List.map json_lift l)
  | _ -> json
     (* raise (Conversion("json_lift: this JSON is not good: "^JSON.to_string json)) *)

module Polish = struct

  let arity = ref "#"

  let of_sexp sexp =
    let mk_token s =
      if not (PPX_Sexp.is_atom s)
      then PPX_Sexp.throw ~who:"Polish.of_sexp/mktoken" sexp;
      let constructor = PPX_Sexp.get_cst ~who:"Polish.of_sexp/mktoken/get_cst" s in
      if not !PPX_Sexp.print_types
      then constructor
      else
        let gram_type = PPX_Sexp.get_type ~who:"Polish.of_sexp/mktoken/gram_type" s in
        constructor^(!arity)^gram_type
    in
    let rec aux accu = function
      | [] -> accu

      | s::totreat when PPX_Sexp.is_atom s ->
        aux ((mk_token s)::accu) totreat

      | (Sexp.List(s::l))::totreat when PPX_Sexp.is_atom s ->
        let token = mk_token s in
        let token =
          if not !PPX_Sexp.print_types
          then 
            let length = List.length l in
            token^(!arity)^string_of_int length
          else
            let aux' sofar s =
              sofar^(!arity)^(PPX_Sexp.get_type ~who:"Polish.of_sexp/aux'" s)
            in
            List.fold_left aux' token l
        in 
        aux (token::accu) (l @ totreat)
  
      | _ -> PPX_Sexp.throw ~who:"Polish.of_sexp/general" sexp
    in
    aux [] [sexp]

  let to_sexp =
    let rec pop n stack out =
      if n=0 then stack,out
      else match stack with
        | [] -> raise(Conversion("Polish.to_sexp: popping out from the stack more elements than the stack contains (polish notation too short)"))
        | head::tail -> pop (n-1) tail (head::out)
    in
    let rec aux stack = function
      | (s,0)::tail -> aux (Sexp.Atom s::stack) tail
      | (s,i)::tail ->
        let stack,out = pop i stack [] in
        aux (Sexp.List(Sexp.Atom s::List.rev out)::stack) tail
      | [] -> match stack with
        | [sexp] -> sexp
        | [] -> raise(Conversion("Polish.to_sexp: stack is empty, it should contain 1 element, namely the result of the conversion"))
        | _::_::_ -> raise(Conversion("Polish.to_sexp: stack has more than 1 element, while it should only contain the result of the conversion (polish notation too long)"))
    in aux []
    
  let to_string =
    let aux sofar s =
      let ending = if (String.length sofar == 0) then sofar else " "^sofar in
      s^ending
    in
    List.fold_left aux ""

  let of_string ~arity s =
    let rec aux out = function
      | []  -> out
      | [s] when Char.equal ' ' arity -> (s,0)::out
      | s::(i::tail as tail') when Char.equal ' ' arity ->
        begin
          match int_of_string_opt i with
          | Some i -> aux ((s,i)::out) tail
          | None   -> aux ((s,0)::out) tail'
        end
      | s::tail ->
        match String.split_on_char arity s with
        | [a;i] ->
          begin match int_of_string_opt i with
            | Some i -> aux ((a,i)::out) tail
            | None -> raise(Conversion("Polish.of_string: in string token "^s^", "^i^" is supposed to be an integer"))
          end
        | [a]   -> aux ((a,0)::out) tail
        | _ -> raise(Conversion("Polish.of_string: too many arity symbols in string "^s))
    in
    aux [] (String.split_on_char ' ' s)
    
end

(* let check ty_to_yojson ty_of_yojson sexp_of_ty ty_of_sexp t = 
 *   let json, sexp = ty_to_yojson t, sexp_of_ty t in
 *   (\* print_endline(Sexp.to_string sexp);
 *    * print_endline(JSON.to_string json);
 *    * print_endline(JSON.to_string(json_of_sexp sexp)); *\)
 *   let polish = Polish.of_sexp sexp in
 *   let polish_string = Polish.to_string ~arity:" " polish in
 *   assert(polish = Polish.of_string ~arity:' ' polish_string);
 *   (\* print_endline(Sexp.to_string(Polish.to_sexp polish));
 *    * print_endline(polish_string); *\)
 *   assert(sexp = Polish.to_sexp polish);
 *   assert(json = json_of_sexp sexp);
 *   (\* assert(sexp_of_json json = sexp); *\)
 *   let rjson, rsexp = exn ty_of_yojson json, ty_of_sexp sexp in
 *   assert(rjson = t && rsexp = t) *)

(*********************)
(* lists and options *)

let typestring_bool = "bool"
let typestring_int = "int"
let typestring_list str = "list<"^str^">"
let typestring_option str = "option<"^str^">"

let sexp_of_bool b =
  let c = if b then "true" else "false" in
  PPX_Sexp.constructor c typestring_bool

let sexp_of_int i =
  let c = string_of_int i in
  PPX_Sexp.constructor c typestring_int

let liststring c str = PPX_Sexp.constructor c (typestring_list str)
let optstring c str = PPX_Sexp.constructor c (typestring_option str)

let rec sexp_of_list (arg,str) l =
  if List.length l = 0 then liststring "Nil" str
  else Sexp.List((liststring "List" str) :: List.map arg l)

let sexp_of_option (arg,str) = function
  | None   -> optstring "None" str
  | Some a -> Sexp.List[optstring "Some" str; arg a]

let rec list_of_sexp arg = function
  | Sexp.Atom "Nil" -> []
  | Sexp.List(Sexp.Atom "List"::(_::_ as l)) -> List.map arg l
  | sexp -> raise(Conversion("list_of_sexp: S-expression "^Sexp.to_string sexp^" does not encode a list"))

let option_of_sexp arg = function
  | Sexp.Atom "None" -> None
  | Sexp.List[Sexp.Atom "Some";a] -> Some(arg a)
  | sexp -> raise(Conversion("list_of_sexp: S-expression "^Sexp.to_string sexp^" does not encode an option"))


(******************)
(* For random AST *)
    
module PPX_Random = struct
  type state = { depth : int; rstate : Random.State.t }
  type 'a t = state -> 'a
  let deepen state = { state with depth = state.depth+1 }
  let case max state = Random.State.int state.rstate max
  let case_30b state = Random.State.bits state.rstate
  let counter = ref 0
  let init () = counter := 0; {depth = 0; rstate = Random.State.make_self_init()}
end

let depth state = Float.of_int(state.PPX_Random.depth+1)

let random_int _  = Random.int 10
let random_bool _ = Random.bool ()

let random_ascii _ =
  let max = 127-32 in
  Char.chr((Random.int max) + 32)

let random_ascii_pair _ =
  let max = 127-32 in
  let a,b = Random.int max, Random.int max in
  if a < b then Char.chr(a+32), Char.chr(b+32)
  else Char.chr(b+32), Char.chr(a+32)

let random_option ?(p=0.5) random_arg state =
  let b = Random.float 1. in
  if Float.(b <= p) then None else Some(random_arg state)

let ( +? ) random_arg w = random_option ~p:w random_arg

let ( *? ) random_arg1 random_arg2 state = random_arg1 state, random_arg2 state

let random_list ?(min=1) ?max ?(empty=0.5) random_arg state =
  let rec aux ?length i accu =
    if i < min ||
        match length with
        | Some l when i < l -> true
        | None when Float.(Random.float 1. > empty) -> true
        | _ -> false
    then aux ?length (i+1) ((random_arg state)::accu)
    else accu
  in
  match max with
  | None   -> aux 0 []
  | Some m -> aux ~length:((Random.int (m + 1 - min) + min)) 0 []

let string_of_char = String.make 1
let random_string state =
  let list = random_list ~min:0 ~max:20 random_ascii state in
  String.escaped(List.fold_right (fun a s -> (string_of_char a)^s) list "")

type 'a distribution = ('a*float) list

(****************)
(* For printing *)

type print       = Format.formatter ->  unit
type 'b pp       = Format.formatter -> 'b -> unit
type ('a,'b) spp = 'a -> 'b pp

let pp_string s fmt () = fprintf fmt s

let pp_list ?(sep=",") ?last pp_arg fmt l =
  let rec aux fmt l =
    match l, last with
    | [], _  -> ()
    | [a], _ -> pp_arg fmt a
    | [a;b], Some last -> fprintf fmt "%a%s %a" pp_arg a last pp_arg b
    | a::tail, _ -> fprintf fmt "%a%s %a" pp_arg a sep aux tail
  in
  aux fmt l  

let pick l _ =
  let sum = List.fold_right (fun (_,a) sofar -> a+sofar) l 0 in
  let rec aux n = function
    | [] -> raise(Conversion("No natural language rendering left to pick from"))
    | (s,i)::tail -> if (n < i) then s else aux (n-i) tail
  in aux (Random.int sum) l

let choose l = pick l () |> Lazy.force

let toString a =
  let buf = Buffer.create 255 in
  let fmt = Format.formatter_of_buffer buf in
  a (Format.fprintf fmt);
  Format.fprintf fmt "%!";
  Buffer.contents buf

let stringOf f a = toString (fun p->p "%a" f a)

(*************************************************)
(* Small extension of the standard Result module *)

module Result = struct
  include Result
  let of_option ~error = function
    | None -> Error error
    | Some b -> Ok b
  let bind a f = match a with
    | Ok a -> f a
    | Error b -> Error b
  let map a f = match a with
    | Ok a -> Ok(f a)
    | Error b -> Error b
end

let (>>=) = Result.bind
let (>>|) = Result.map

(***********************)
(* Module for entities *)

module Entity = struct

  let one_entity_kind = ref false (* Is there just one entity kind? *)
  let warnings : [`NoSubst of string] list ref = ref [] (* While parsing a json, are we seeing any warning? *)
  let strict = ref 1.5
  
  type 'a t = {
    kind : 'a option;
    counter : int;
    substitution : string option
  }

  let typestring arg = "entity<"^arg^">"

  let pp pp_arg fmt e =
    let pp_kind fmt = function
      | Some e when not !one_entity_kind -> fprintf fmt "%a" pp_arg e
      | _ -> fprintf fmt "Entity"
    in
    match e.substitution with
    | Some nl -> fprintf fmt "%a_%i{%s}" pp_kind e.kind e.counter nl
    | None -> fprintf fmt "%a_%i" pp_kind e.kind e.counter

  let random random_arg s =
    incr PPX_Random.counter;
    { kind = Some(random_arg s);
      counter = !PPX_Random.counter;
      substitution = None }

  let to_yojson arg e =
    let base =
      match e.kind with
      | Some k when not !one_entity_kind ->
        begin match arg k with
          | `List[`String s] -> `String(s^"_"^string_of_int e.counter)
          | _ -> raise(Conversion("Entity.to_yojson on "^JSON.to_string(arg k)))
        end
      | _ -> `String("Entity_"^string_of_int e.counter)
    in
    match e.substitution with
    | Some entity -> `List[base; `String entity]
    | None -> `List[base]

  let sexp_of (arg,argt) e =
    let base =
      match e.kind with
      | Some k when not !one_entity_kind ->
        let cst = PPX_Sexp.get_cst ~who:"Entity.sexp_of" (arg k) in
        PPX_Sexp.constructor (cst^"_"^string_of_int e.counter) (typestring argt)
      | _ -> Sexp.Atom("Entity_"^string_of_int e.counter)
    in
    match e.substitution with
    | Some entity -> Sexp.List[base;Sexp.Atom entity]
    | None -> base 

  let to_id l =
    let rec aux ?accu l =
      match l, accu with
      | [], _ 
      | [(_)], None   -> Error "not good"
      | [i], Some s -> 
        Result.of_option
          ~error:(i^" should be an integer")
          (int_of_string_opt i) >>| fun i -> (s,i)
      | accu::tail, None   -> aux ~accu tail
      | accu::tail, Some s -> aux ~accu:(s^"_"^accu) tail
    in
    String.split_on_char '_' l |> aux 

  let entity_mk s ?(kind=None) ?(counter=0) ?(substitution=None) () =
    (match substitution with None -> warnings := (`NoSubst s)::!warnings | Some _ -> ());
    Result.Ok { kind; counter; substitution }
      
  let rec of_yojson arg = function
    | `String _ as json -> of_yojson arg (`List[json])
    | `List(`String s::nl) ->
      let substitution = match nl with
        | [`String nl]
        | [`List[`String nl]] -> Some nl
        | _ -> None
      in
      begin
        match to_id s with
        | Result.Ok(e,counter) ->
          begin match arg(`List [`String e]) with
            | Result.Ok k -> entity_mk s ~kind:(Some k) ~counter ~substitution ()
            | Result.Error _ -> entity_mk s ~counter ~substitution ()
          end
        | Result.Error _ -> entity_mk s ~substitution ()
      end
    | `List[s] -> of_yojson arg s
    | json -> Result.Error("JSON for an entity cannot be "^JSON.to_string json)

  let t_of_sexp arg = function
    | Sexp.Atom s ->
      let s,counter = exn to_id s in
      if s = "Entity" then { kind = None; counter; substitution = None }
      else { kind = Some(arg (Sexp.Atom s)); counter; substitution = None }
    | Sexp.List[Sexp.Atom s;Sexp.Atom nl] ->
      let s,counter = exn to_id s in
      if s = "Entity" then { kind = None; counter; substitution = Some nl }
      else { kind = Some(arg (Sexp.Atom s)); counter; substitution = Some nl }
    | Sexp.List _ as sexp -> raise(Conversion("Entity.t_of_sexp: list S-expresion "^Sexp.to_string sexp^" cannot be an entity"))

  let pick l = pick(List.map (fun (x,i) -> x, Float.(to_int(pow i !strict))) l)

end


