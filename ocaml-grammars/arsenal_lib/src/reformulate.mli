open Arsenal_lib
   
val main : port:int -> (JSON.t -> ('a, string) Result.result) -> 'a pp -> unit Lwt.t
val port : int ref
val description : string -> string
