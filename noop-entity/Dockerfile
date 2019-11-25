FROM openjdk:8-alpine

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
COPY target/noop-entity-0.1.0-SNAPSHOT-standalone.jar /opt/compojure/noop-entity.jar

WORKDIR /opt/compojure

CMD PORT=${HTTP_PORT} \
  java -jar /opt/compojure/noop-entity.jar
