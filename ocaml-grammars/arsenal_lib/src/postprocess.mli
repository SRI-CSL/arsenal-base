open Arsenal_lib
   
val main : port:int
           -> 'a PPX_Serialise.t
           -> ( ?options: (string*JSON.t) list -> id:JSON.t -> 'a -> JSON.t)
           -> unit Lwt.t
val port : int ref
val description : string -> string
