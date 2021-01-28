open Containers
open Arsenal_lib
open Sexplib
   
open Lwt.Infix
open Cohttp_lwt
open Cohttp_lwt_unix

let good_object reformulations = `Assoc ["result", `List reformulations]

let error_object ?(id=`Null) ?(json=`Null) message =
  `Assoc ["id",id; "error",`String message; "json",json ]

let exc s json = raise(Conversion(s^" "^JSON.to_string json))
                  
module Dictionary = Hashtbl.Make(String)

let rec json2sexp dictionary = function
  | `String key when Dictionary.mem dictionary key
    -> Sexp.List[Sexp.Atom key; Sexp.Atom(Dictionary.find dictionary key)]
  | `String s -> Sexp.Atom s
  | `List l   -> Sexp.List(List.map (json2sexp dictionary) l)
  | json      -> exc "The following JSON is not a Sexp:" json

let reformulate cst_conv cst_process json_string =
  try
    let treat_one json =
      let id = JSON.Util.member "id" json in
      let cst = JSON.Util.member "cst" json in
      try
        print_endline("JSON: "^JSON.to_string cst);
        let options = match JSON.Util.member "options" json with
          | `Null    -> None
          | `Assoc l -> Some l
          | json -> exc "The following JSON is not a list of options:" json
        in
        let dictionary = Dictionary.create 10 in
        let aux (key, value) = match value with
          | `String s -> Dictionary.add dictionary key s
          | json -> exc "The substitution for an entity should be a string, not:" json
        in
        let () = match JSON.Util.member "substitutions" json with
          | `Assoc l -> List.iter aux l
          | json -> exc "The substitution should be a JSON dictionary, not:" json
        in
        json2sexp dictionary cst |> cst_conv.PPX_Serialise.of_sexp |> cst_process ?options ~id
      with
      | Conversion error ->
        error_object ~id ~json ("Problem with conversion while reading: "^error)
    in
    json_string
    |> JSON.from_string
    |> JSON.Util.member "sentences"
    |> JSON.Util.to_list
    |> List.map treat_one
    |> good_object
  with
  | Yojson.Json_error error ->
    error_object ("Problem with string->JSON conversion: "^error)
  | JSON.Util.Type_error(error,json)
  | JSON.Util.Undefined(error,json) ->
    error_object ~json ("Problem extracting info from JSON: "^error)


(* Apply the [Webmachine.Make] functor to the Async-based IO module exported by
 * cohttp. For added convenience, include the [Rd] module as well so you don't
 * have to go reaching into multiple modules to access request-related
 * information. *)
module Wm = struct
  module Rd = Webmachine.Rd
  module UnixClock = struct
    let now = fun () -> int_of_float (Unix.gettimeofday ())
  end
  include Webmachine.Make(Cohttp_lwt_unix__Io)(UnixClock)
end

(* Create a new class that inherits from [Wm.resource] and provides
 * implementations for its two virtual methods, and overrides some of its default methods.
 *)
class hello cst_conv cst_process = object(self)
  inherit [Body.t] Wm.resource

  (* Only allow POST requests to this resource *)
  method allowed_methods rd =
    Wm.continue [`POST] rd

  (* Setup the resource to handle multiple content-types. Webmachine will
   * perform content negotiation as described in RFC 7231:
   *
   *   https://tools.ietf.org/html/rfc7231#section-5.3.2
   *
   * Content negotiation can be a complex process. Hoever for simple Accept
   * headers its fairly straightforward. Here's what content negotiation will
   * produce in some of these simple cases:
   *
   *     Accept             | Called method
   *   ---------------------+----------------
   *     "text/plain"       | self#to_text
   *     "text/html"        | self#to_html
   *     "text/*"           | self#to_html
   *     "application/json" | self#to_json
   *     "application/*"    | self#to_json
   *     "*/*"              | self#to_html
   *)
  method content_types_provided rd =
    Wm.continue [
      ("text/html"       , self#to_html);
      ("text/plain"      , self#to_text);
      ("application/json", self#to_json);
    ] rd

  method content_types_accepted rd =
    Wm.continue [] rd

  (* Returns an html-based representation of the resource *)
  method private to_html rd =
    let body = Printf.sprintf "<html><body>to_html</body></html>" in
    Wm.continue (`String body) rd

  (* Returns a plaintext representation of the resource *)
  method private to_text rd =
    let text= Printf.sprintf "to_text" in
    Wm.continue (`String text) rd

  (* Returns a json representation of the resource *)
  method private to_json rd =
    let json = Printf.sprintf "{\"output\" : \"to_json\"}" in
    Wm.continue (`String json) rd

  (* Process POST request *)
  method process_post rd =
    Body.to_string rd.req_body >>= fun json_string ->
    let resp_body = json_string
                    |> reformulate cst_conv cst_process
                    |> JSON.to_string
                    |> Body.of_string
    in
    Wm.continue true { rd with resp_body }

end

let main ~port cst_conv cst_process =
  (* The route table. Both routes use the [hello] resource defined above.
   * However, the second one containes the [:what] wildcard in the path. The
   * value of that wildcard can be accessed in the resource by calling
   *
   *   [Wm.Rd.lookup_path_info "what" rd]
  *)
  let routes = [
    ("/"           , fun () -> new hello cst_conv cst_process);
  ] in
  let callback (_ch,_conn) request body =
    let open Cohttp in
    (* Perform route dispatch. If [None] is returned, then the URI path did not
     * match any of the route patterns. In this case the server should return a
     * 404 [`Not_found]. *)
    Wm.dispatch' routes ~body ~request
    >|= begin function
      | None        -> (`Not_found, Header.init (), `String "Not found", [])
      | Some result -> result
    end
    >>= fun (status, headers, body, path) ->
    (* If you'd like to see the path that the request took through the
     * decision diagram, then run this example with the [DEBUG_PATH]
     * environment variable set. This should suffice:
     *
     *  [$ DEBUG_PATH= ./hello_async.native]
     *
    *)
    let path =
      match Sys.getenv "DEBUG_PATH" with
      | _ -> Printf.sprintf " - %s" (String.concat ", " path)
      | exception Not_found   -> ""
    in
    Printf.eprintf "%d - %s %s%s"
      (Code.code_of_status status)
      (Code.string_of_method (Request.meth request))
      (Uri.path (Request.uri request))
      path;
    (* Finally, send the response to the client *)
    Server.respond ~headers ~body ~status ()
  in
  (* Create the server and handle requests with the function defined above. Try
   * it out with some of these curl commands:
   *
   *   [curl -H"Accept:text/html" "http://localhost:8080"]
   *   [curl -H"Accept:text/plain" "http://localhost:8080"]
   *   [curl -H"Accept:application/json" "http://localhost:8080"]
  *)
  let conn_closed (ch,_conn) =
    Printf.printf "connection %s closed\n%!"
      (Sexplib.Sexp.to_string_hum (Conduit_lwt_unix.sexp_of_flow ch))
  in
  let config = Server.make ~callback ~conn_closed () in
  Printf.eprintf "hello_lwt: listening on localhost:%d\n%!" port;
  Server.create  ~mode:(`TCP(`Port port)) config

let port = ref 8080

let description arg =
  "This is an ocaml reformulator for "^arg^" represented in JSON. The command takes the port number to listen to as optional argument (8080 is the default)."
