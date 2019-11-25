(ns noop-entity.handler
  (:require [compojure.core :refer :all]
            [compojure.route :as route]
            [noop-entity.core :as noop]
            [ring.util.response :refer [response]]
            [ring.util.request :refer [body-string]]
            [ring.middleware.json :refer [wrap-json-params wrap-json-response wrap-json-body]]
            [ring.middleware.defaults :refer [wrap-defaults api-defaults]]))

(defroutes app-routes
  (GET "/" [] "No-op Entity Processor is up and running...")
  
  (POST "/process" [& params]
    (response (noop/process-text params)))
  
  (POST "/process_all" [& params]
    (response (noop/process-text-all (:sentences params))))

  (route/not-found "Not Found"))

(defn allow-cross-origin
  "middleware function to allow cross origin"
  [handler]
  (fn [request]
    (-> (handler request)
        (assoc-in [:headers "Access-Control-Allow-Origin"]  "*")
        (assoc-in [:headers "Access-Control-Allow-Methods"] "GET,PUT,POST,DELETE,OPTIONS")
        (assoc-in [:headers "Access-Control-Allow-Headers"] "X-Requested-With,Content-Type,Cache-Control"))))
  
(def app
  (-> app-routes
      (wrap-defaults api-defaults)
      (wrap-json-params)
      (wrap-json-response)
      (allow-cross-origin)))