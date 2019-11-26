# First stage, build the JS
FROM alpine:latest AS regexp-ui-builder
RUN apk add --update nodejs npm yarn
ADD /package.json /opt/regexp/
ADD /yarn.lock    /opt/regexp/
WORKDIR /opt/regexp
RUN yarn -y install
ADD /public       /opt/regexp/public
ADD /src          /opt/regexp/src
RUN yarn build

# Second stage, copy the build files to a fresh nginx image
FROM nginx:latest
COPY docker/proxy.conf /etc/nginx/
COPY docker/nginx.conf /etc/nginx/
COPY docker/default.conf /etc/nginx/conf.d/
COPY --from=regexp-ui-builder /opt/regexp/build /usr/share/nginx/html
CMD ["nginx", "-g", "daemon off;"]
