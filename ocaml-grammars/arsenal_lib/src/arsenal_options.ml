open Containers
open Sexplib
open Arg
open Arsenal_lib

module JSON = Yojson.Safe;;

(************************)
(* Command line options *)

let check_opt  = ref false (* Check outputs *)
let print_nl   = ref true  (* Output Natural Language by default *)
let print_json = ref false (* Do not output json by default *)
let print_sexp = ref false (* Do not output S-expression by default *)
let print_polish = ref false (* Do not output polish notation by default *)
let howmany = ref 1        (* Default number of example to generate is 1 *)
let raw_json = ref false   (* Whether json should just be like Sexp *)
let injectivity = ref false (* Whether to check that the pretty-printing function is injective -- requires saving in memory all instances produced *)
let no_duplicates = ref false (* Whether to ensure there are no duplicates -- requires saving in memory all instances produced *)
let args : string list ref = ref [] (* Where the command-line argument will be stacked *)

let options =
  [
    ("-n", Int(fun i -> howmany := i), "\thow many instances to generate (default is 1)");
    ("-json", Set print_json, "\tprint json output");
    ("-sexp", Set print_sexp, "\tprint S-expression output");
    ("-polish", Set print_polish, "\tprint polish output");
    ("-raw-json", Set raw_json, "\tmaximal Produce raw JSONs (i.e. JSON versions of S-expressions) (default is false)");
    ("-arity_sep", String(fun s -> Polish.arity := s), "\tin Polish notation, specify separator between token and its arity (default is '#')");
    ("-one-entity", Set Entity.one_entity_kind, "\tin natural language, only use 1 kind of entities");
    ("-types", Set PPX_Serialise.print_types, "\tdisplay types in generated data");
    ("-no-nl", Clear print_nl, "\tprint disable natural language output");
    ("-no-duplicates", Set no_duplicates, "\tensure there are no duplicates (default is false) -- saves generated instances in memory");
    ("-check-injectivity", Set injectivity, "\tcheck there are no two ASTs for the same NL (default is false) -- saves generated instances in memory");
    (* ("-check", Set check_opt, "\tcheck output"); *)
    ("-strict-entities", Float(fun f -> Entity.strict := f), "\thow strict entity kinds should be considered (0. : they are ignored; +infty: very strictly enforced)");
  ]

module STbl = CCHashtbl.Make(String)


module Generate_runtime = struct

  type 'a t = {
    random    : PPX_Random.state -> 'a;
    pp        : 'a pp;
    to_json   : 'a -> JSON.t;
    to_sexp   : 'a -> Sexp.t;
  }

  exception NonInjective of { nl      : string;
                              ast_old : Sexp.t ;
                              ast_new : Sexp.t }

  let generate grammar n =
    let memo =
      if !injectivity || !no_duplicates
      then STbl.create (2*n)
      else STbl.create 1
    in
    let print_tab s = print_string ("\t"^s) in
    let sofar = ref 0 in
    while (!sofar < n) do
      let t = PPX_Random.init() |> grammar.random in
      Entity.init();
      (* if !check_opt  then check sentence_to_yojson sentence_of_yojson sexp_of_sentence sentence_of_sexp t; *)
      let nl =
        if !print_nl then t |> grammar.pp |> toString
        else "No NL constructed"
      in
      let sexp () =
        if !print_sexp || !raw_json || !print_polish then t |> grammar.to_sexp
        else Atom "No sexp constructed"
      in
      let go_ahead sexp =
        if !print_nl     then print_string nl;
        if !print_json   then t |> grammar.to_json |> JSON.to_string     |> print_tab;
        if !print_polish then sexp |> Polish.of_sexp |> Polish.to_string |> print_tab;
        if !raw_json     then sexp |> sexp2json |> JSON.to_string        |> print_tab;
        if !print_sexp   then sexp |> Sexp.to_string                     |> print_tab;
        print_endline "";
        incr sofar;
      in
      if !injectivity || !no_duplicates
      then
        match STbl.find_opt memo nl with
        | Some sexp' ->
           if not !injectivity then () (* not checking injectivity means we reject duplicates *) 
           else (* checking injectivity *)
             let sexp = sexp () in
             if not (Sexp.equal sexp sexp')
             then
               begin
                 Format.(fprintf stderr "@[<v>Natural language@,  @[%s@]@,maps to@,  @[<v>%a@]@,and to@,  @[<v>%a@]@,@]" nl Sexplib.Sexp.pp sexp' Sexplib.Sexp.pp sexp);
                 raise (NonInjective { nl ; ast_old = sexp' ; ast_new = sexp })
               end;
             if not !no_duplicates then go_ahead sexp
        | None -> let sexp = sexp () in STbl.add memo nl sexp; go_ahead sexp
      else 
        go_ahead (sexp())
    done

end
