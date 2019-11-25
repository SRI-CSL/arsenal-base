(ns noop-entity.core
    (:require [environ.core :refer [env]]
              [clojure.string :as str]
              [clojure.pprint :as pprint]))

(defn process-text [{:keys [text id]}]
    (println "Processing text" text)
    { :id id 
      :orig-text text 
      :new-text text
      :substitutions {} })

(defn process-text-all [sentences]
    { :sentences (map process-text sentences) })