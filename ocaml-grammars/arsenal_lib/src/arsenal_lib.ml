open Containers
open Format

open Sexplib

module JSON = Yojson.Safe

(************************************************************)
(* Conversions between json, S-expressions, Polish notation *)

exception Conversion of string

module PPX_Sexp = struct  
  let print_types = ref false (* Does not print types in S-expressions *)

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

module JSONindex = struct
  let index = ref []
  module JSONhashtbl = Hashtbl.Make(String)
  let marked = JSONhashtbl.create 100
  type t = (string * JSON.t) list ref
  let mem s  = JSONhashtbl.mem marked s
  let find s = !(JSONhashtbl.find marked s)
  let mark s : t = 
    let result = ref [] in
    JSONhashtbl.add marked s result;
    result
  let add (mark : t) l =
    mark := l;
    List.iter (fun json -> index := json::!index) l
  let out ~id ~description =
    let toptype = match !index with
      | [] -> failwith "No entry point given"
      | (toptype,_)::_ -> toptype
    in
    `Assoc [
        "$schema", `String "http://json-schema.org/schema#";
        "$id",     `String id;
        "description", `String description;
        "$ref",     `String("#/definitions/"^ toptype);
        "definitions", `Assoc (!index |> List.rev )
      ]
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
  | `Assoc _  ->
     let node = JSON.Util.member "node" json  in
     let args =
       match JSON.Util.member "subtrees" json with
       | `List l -> l
       | `Null -> []
       | a -> [a]
     in
     begin
       match node with
       | `String "True" | `String "true" | `Bool true -> `Bool true
       | `String "False" | `String "false" | `Bool false -> `Bool false
       | `String "None" -> `Null
       | `String "Some" -> json_clean (List.hd args)
       | `String "Nil" | `String "List" -> `List(List.map json_clean args)
       | `String _ | `List _ | `Assoc _ -> `List(node::List.map json_clean args)
       | `Null ->
          begin match JSON.Util.member "subtrees" json with
          | `List args -> `List(List.map json_clean args)
          | `Null -> `Null
          | arg -> json_clean arg
          end
       | _ -> raise (Conversion("json_clean: the \"node\" field in this JSON should be a string or an entity: "^JSON.to_string json))
     end
  | `Null | `Bool _ -> json
  | `List l   -> `List(List.map json_clean l)
  | _ -> raise (Conversion("clean_types: this JSON is not good: "^JSON.to_string json))

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
      let ending = if (String.length sofar = 0) then sofar else " "^sofar in
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
let typestring_list str = "list("^str^")"
let typestring_option str = "option("^str^")"
let json_desc_list _ () = ()
let json_desc_option _ () = ()

let sexp_of_bool b =
  let c = if b then "true" else "false" in
  PPX_Sexp.constructor c typestring_bool

let sexp_of_int i =
  let c = string_of_int i in
  PPX_Sexp.constructor c typestring_int

let liststring c str = PPX_Sexp.constructor c (typestring_list str)
let optstring c str = PPX_Sexp.constructor c (typestring_option str)

let sexp_of_list (arg,str) l =
  if List.length l = 0 then liststring "Nil" str
  else Sexp.List((liststring "List" str) :: List.map arg l)

let sexp_of_option (arg,str) = function
  | None   -> optstring "None" str
  | Some a -> Sexp.List[optstring "Some" str; arg a]

let list_of_sexp arg = function
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

let random_int state = Random.int 10 state.PPX_Random.rstate
let random_bool _ = Random.bool ()

let random_ascii state =
  let max = 127-32 in
  Char.chr((Random.int max state.PPX_Random.rstate) + 32)

let random_ascii_pair state =
  let max = 127-32 in
  let a,b = Random.int max state.PPX_Random.rstate, Random.int max state.PPX_Random.rstate in
  if a < b then Char.chr(a+32), Char.chr(b+32)
  else Char.chr(b+32), Char.chr(a+32)

let random_option ?(p=0.5) random_arg state =
  let b = Random.float 1. state.PPX_Random.rstate in
  if Float.(b <= p) then None else Some(random_arg state)

let ( +? ) random_arg w = random_option ~p:w random_arg

let ( *? ) random_arg1 random_arg2 state = random_arg1 state, random_arg2 state

let random_list ?(min=1) ?max ?(empty=0.5) random_arg state =
  let rec aux ?length i accu =
    if i < min ||
        match length with
        | Some l when i < l -> true
        | None when Float.(Random.float 1. state.PPX_Random.rstate > empty) -> true
        | _ -> false
    then aux ?length (i+1) ((random_arg state)::accu)
    else accu
  in
  match max with
  | None   -> aux 0 []
  | Some m -> aux ~length:((Random.int (m + 1 - min) state.PPX_Random.rstate + min)) 0 []

let string_of_char = String.make 1
let random_string state =
  let list = random_list ~min:0 ~max:20 random_ascii state in
  String.escaped(List.fold_right (fun a s -> (string_of_char a)^s) list "")

type 'a distribution = ('a*float) list

(****************)
(* For printing *)

type print = Format.formatter ->  unit
type 'a pp = 'a -> print

let return s fmt = Format.fprintf fmt "%s" s
let noop     _fmt = ()
let (^^) pp1 pp2 fmt = pp1 fmt; pp2 fmt
                 
type 'a formatted =
  | F : ('a , Format.formatter, unit) format -> 'a formatted
  | FormatApply : ('a -> 'b) formatted * 'a -> 'b formatted
  | Noop : unit formatted

let rec print : type a. a formatted -> formatter -> a = function
  | F s    -> fun fmt -> Format.fprintf fmt s
  | FormatApply(a,b) -> fun fmt -> print a fmt b
  | Noop -> noop

let (//) a b = FormatApply(a,b)

let pick l =
  let sum = List.fold_right (fun (_,a) sofar -> a+sofar) l 0 in
  let rec aux n = function
    | [] -> raise(Conversion("No natural language rendering left to pick from"))
    | (s,i)::tail -> if (n < i) then s else aux (n-i) tail
  in
  let state = PPX_Random.init() in
  aux (Random.int sum state.rstate) l

let toString a =
  let buf = Buffer.create 255 in
  let fmt = Format.formatter_of_buffer buf in
  let () = a fmt in
  Format.fprintf fmt "%!";
  Buffer.contents buf

let pp_list ?(sep=",") ?last pp_arg l =
  let rec aux l =
    match l, last with
    | [], _  -> noop
    | [a], _ -> pp_arg a
    | [a;b], Some last -> F "%t%s %t" // pp_arg a // last // pp_arg b |> print
    | a::tail, _ -> F "%t%s %t" // pp_arg a // sep // aux tail |> print
  in
  aux l

(* Easy choose *)
(* For lists of strings *)
let (!!)  l fmt = (pick(List.map (fun s -> s, 1) l) |> return) fmt
let (!!!) l fmt = (pick l |> return) fmt
(* For lists of %t functions *)
let (!~)  l fmt = pick(List.map (fun s -> s,1) l) fmt
let (!~~) l fmt = pick l fmt
(* For lists of anything *)
let (!&)  l _   = pick(List.map (fun s -> s,1) l)
let (!&&) l _   = pick l
(* For lists of formatted *)
let (!?)  l fmt = (pick(List.map (fun s -> s,1) l) |> print) fmt
let (!??) l fmt = (pick l |> print) fmt

(* Easy weights *)
let (?~) b   = if b then 1 else 0
let (~?) b   = if b then 0 else 1
let (??~) o  = match o with Some _ -> 1 | None -> 0
let (~??) o  = 1 - ??~o
let (++) w1 w2 = 1 - (w1 * w2)

(* Easy extension of pp function to option type *)
let (+?) pp1 pp2  = function Some x -> pp1 x | None -> pp2
let (?+) pp_arg = pp_arg +? noop
let (?++) = function Some _ -> true | None -> false

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
    kind         : 'a option;
    counter      : int;
    substitution : string option
  }

  let typestring arg = "entity("^arg^")"
  let json_desc arg () =
    let mark = JSONindex.mark arg in
    JSONindex.add mark
      [ typestring arg,
        `Assoc [ "type",       `String "object";
                 "required",   `List [`String "counter"]; 
                 "properties", `Assoc [
                                   "entity",       `Assoc ["type", `String "boolean"];
                                   "kind",         `Assoc ["type", `String "string"];
                                   "counter",      `Assoc ["type", `String "integer"];
                                   "substitution", `Assoc ["type", `String "string"]
                                 ]
          ]
      ]

  let pp pp_arg e =
    let pp_kind = function
      | Some e when not !one_entity_kind -> F "%a" // pp_arg // e |> print
      | _ -> return "Entity"
    in
    match e.substitution with
    | Some nl -> F "%t_%i{%s}" // pp_kind e.kind // e.counter // nl |> print
    | None    -> F "%t_%i" // pp_kind e.kind // e.counter |> print

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

  let entity_mk s ?(kind=None) ?(counter=0) ?(substitution=None) () =
    (match substitution with None -> warnings := (`NoSubst s)::!warnings | Some _ -> ());
    Result.Ok { kind; counter; substitution }

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

  let of_yojson_aux arg placeholder substitution = 
    match to_id placeholder with
    | Result.Ok(e,counter) ->
       begin match arg(`List [`String e]) with
       | Result.Ok k -> entity_mk placeholder ~kind:(Some k) ~counter ~substitution ()
       | Result.Error _ -> entity_mk placeholder ~counter ~substitution ()
       end
    | Result.Error _ -> entity_mk placeholder ~substitution ()
    
  let rec of_yojson arg = function
    | `String _ as json -> of_yojson arg (`List[json])
    | `List(`String placeholder::nl) ->
      let substitution = match nl with
        | [`String nl]
        | [`List[`String nl]] -> Some nl
        | _ -> None
      in
      of_yojson_aux arg placeholder substitution
    | `List[s] -> of_yojson arg s
    | `Assoc _ as json ->
       begin
         match JSON.Util.member "placeholder" json with
         | `String placeholder ->
            let substitution = match JSON.Util.member "text" json with
              | `String text -> Some text
              | _ -> None
            in
            of_yojson_aux arg placeholder substitution
         | _ -> Result.Error("JSON for an entity cannot be "^JSON.to_string json)
       end
    | json -> Result.Error("JSON for an entity cannot be "^JSON.to_string json)

  let t_of_sexp arg = function
    | Sexp.Atom s ->
      let s,counter = exn to_id s in
      if String.equal s "Entity" then { kind = None; counter; substitution = None }
      else { kind = Some(arg (Sexp.Atom s)); counter; substitution = None }
    | Sexp.List[Sexp.Atom s;Sexp.Atom nl] ->
      let s,counter = exn to_id s in
      if String.equal s "Entity" then { kind = None; counter; substitution = Some nl }
      else { kind = Some(arg (Sexp.Atom s)); counter; substitution = Some nl }
    | Sexp.List _ as sexp -> raise(Conversion("Entity.t_of_sexp: list S-expresion "^Sexp.to_string sexp^" cannot be an entity"))

  let pick l _ = pick(List.map (fun (x,i) -> x, Stdlib.Float.(to_int(pow i !strict))) l)

end


