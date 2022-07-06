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
let pretty     = ref false  (* prettyfy JSON and S-expressions *) 
let howmany    = ref 1      (* Default number of example to generate is 1 *)
let raw_json   = ref false  (* Whether json should just be like Sexp *)
let injectivity = ref false (* Whether to check that the pretty-printing function is injective -- requires saving in memory all instances produced *)
let no_duplicates = ref false (* Whether to ensure there are no duplicates -- requires saving in memory all instances produced *)
let args : string list ref = ref [] (* Where the command-line argument will be stacked *)

let options_base =
  [
    ("-hide-entities", Clear Entity.ppkind, "\tif a substitution string is present in entity, just show the string");
    ("-hide-substitutions", Clear Entity.show_substitutions, "\tdo not show substitution strings in entities");
    ("-json", Set print_json, "\tprint json output");
    ("-no-nl", Clear print_nl, "\tprint disable natural language output");
    ("-one-entity", Set Entity.one_entity_kind, "\tin natural language, only use 1 kind of entities");
    ("-path-mode", Int(fun i -> qualify_mode := if i < 0 then None else Some i), "\tmode for displaying paths in contructors and entity kinds: -1 for no path, 0 for whole paths, (positive) i for pruning the first i levels of paths (default is 0)");
    ("-pretty", Set pretty, "\tprint JSON and S-expressions in human-readable form");
    ("-separator", String(fun s -> separator := s), "\tseparator for module names in constructors and entity kinds (default is \".\")");
    ("-sexp", Set print_sexp, "\tprint S-expression output");
    ("-short", Unit(fun () -> qualify_mode := None), "\tequivalent to no path (-path-mode \"-1\") in contructors and entity kinds");
    ("-type-arg", Tuple [String (fun pre -> str_arg := pre, snd !str_arg); String (fun post -> str_arg := fst !str_arg, post)] , "\t pre- and post- delimiters for type arguments (default is \"(\" \")\")");
    ("-verb", Int(fun i -> verb := i), "\tverbosity level (default is 0)");
  ]

let options =
  [
    ("-arity_sep", String(fun s -> Polish.arity := s), "\tin Polish notation, specify separator between token and its arity (default is '#')");
    (* ("-check", Set check_opt, "\tcheck output"); *)
    ("-check-injectivity", Set injectivity, "\tcheck there are no two ASTs for the same NL (default is false) -- saves generated instances in memory");
    ("-deterministic", Set deterministic, "\tDeterministic NL generation (default is false)");
    ("-n", Int(fun i -> howmany := i), "\thow many instances to generate (default is 1)");
    ("-no-duplicates", Set no_duplicates, "\tensure there are no duplicates (default is false) -- saves generated instances in memory");
    ("-polish", Set print_polish, "\tprint polish output");
    ("-raw-json", Set raw_json, "\tProduce raw JSONs (i.e. JSON versions of S-expressions) (default is false)");
    ("-strict-entities", Float(fun f -> Entity.strict := f), "\thow strict entity kinds should be considered (0. : they are ignored; +infty: very strictly enforced)");
    ("-types", Set PPX_Serialise.print_types, "\tdisplay types in generated data");
  ] @ options_base

exception NonInjective of { nl      : string;
                            ast_old : Sexp.t ;
                            ast_new : Sexp.t }

let generate (About.About{ key ; serialise = { to_json ; to_sexp; _} ; _ }) n =
  let memo =
    if !injectivity || !no_duplicates
    then Stbl.create (2*n)
    else Stbl.create 1
  in
  let print_tab s = print_string ("\t"^s) in
  let sofar = ref 0 in
  let duplicates = ref 0 in
  let pp_json =
    if !pretty then (fun x -> JSON.pretty_to_string x) else (fun x -> JSON.to_string x)
  in
  while (!sofar < n) do
    let t = PPX_Random.init() |> !(TUID.get_random key) in
    Entity.init();
    (* if !check_opt  then check sentence_to_yojson sentence_of_yojson sexp_of_sentence sentence_of_sexp t; *)
    let nl =
      if !print_nl then t |> !(TUID.get_pp key) |> toString
      else "No NL constructed"
    in
    let sexp () =
      if !print_sexp || !raw_json || !print_polish then t |> to_sexp
      else Atom "No sexp constructed"
    in
    let go_ahead sexp =
      if !print_nl     then print_string nl;
      if !print_json   then t    |> to_json        |> pp_json          |> print_tab;
      if !print_polish then sexp |> Polish.of_sexp |> Polish.to_string |> print_tab;
      if !raw_json     then sexp |> sexp2json      |> pp_json          |> print_tab;
      if !print_sexp   then sexp
                            |> Format.sprintf "@,@[<v>%a@]"
                                 Sexp.(if !pretty then pp_hum else pp_mach)
                            |> print_tab;
      print_endline "";
      if !verb > 0 && !sofar mod 1000 = 0 then Format.(fprintf stderr) "@[%i@]@,%!" !sofar;
      incr sofar;
    in
    begin
      if !injectivity || !no_duplicates
      then
        begin
          match Stbl.find_opt memo nl with
          | Some sexp' ->
             incr duplicates;
             if not !injectivity then () (* not checking injectivity means we reject duplicates *) 
             else (* checking injectivity *)
               let sexp = sexp () in
               if not (Sexp.equal sexp sexp')
               then
                 begin
                   Format.(fprintf stderr "@[<v>Natural language@,  @[%s@]@,maps to@,  @[<v>%a@]@,and to@,  @[<v>%a@]@,@]" nl Sexp.pp_hum sexp' Sexp.pp_hum sexp);
                   raise (NonInjective { nl ; ast_old = sexp' ; ast_new = sexp })
                 end;
               if not !no_duplicates then go_ahead sexp
          | None ->
             let sexp = sexp () in Stbl.add memo nl sexp; go_ahead sexp
        end
      else 
        go_ahead (sexp())
    end;
    if !sofar mod 1000 = 0 || (!duplicates > 0 && !duplicates mod 1000 = 0)
    then
      begin
        let dup = if !duplicates > 0 then Format.sprintf " (%d dup)" !duplicates else "" in 
        Format.(fprintf stderr "@[<v>%d%s / %d@]\n%!" !sofar dup n)
      end
  done
