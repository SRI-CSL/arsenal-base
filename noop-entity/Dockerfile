# Stage 1. Build the uberjar using Leiningen
FROM clojure:temurin-8-lein-alpine AS noop-entity-builder
ADD /project.clj /opt/
ADD /src /opt/src
WORKDIR /opt
RUN lein ring uberjar


# Stage 2. Run the uberjar in a fresh Java image
FROM eclipse-temurin:8-alpine
RUN apk --no-cache add shadow

ENV \
HTTP_PORT=${HTTP_PORT:-8080} \
CTAKES_HOME=/opt/ctakes

EXPOSE ${HTTP_PORT}

RUN mkdir -p /opt/compojure

# Create the compojure user and group
RUN groupadd -r compojure -g 433 && useradd -u 431 -r -g compojure -d /opt/compojure -s /sbin/nologin -c "compojure user" compojure && \
  chown -R compojure:compojure /opt

USER compojure

# Add the service uberjar
COPY --from=noop-entity-builder /opt/target/noop-entity-0.1.0-SNAPSHOT-standalone.jar /opt/compojure/noop-entity.jar

WORKDIR /opt/compojure

CMD PORT=${HTTP_PORT} \
  java -jar /opt/compojure/noop-entity.jar
