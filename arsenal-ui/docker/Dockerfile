# First stage, build the JS
FROM alpine:latest AS arsenal-ui-builder
RUN apk add --update nodejs npm yarn
ADD /package.json /opt/arsenal/
ADD /yarn.lock    /opt/arsenal/
WORKDIR /opt/arsenal
RUN yarn config set strict-ssl false
RUN yarn -y install
ADD /public       /opt/arsenal/public
ADD /src          /opt/arsenal/src
RUN yarn cache clean
RUN yarn build

# Second stage, copy the build files to a fresh nginx image
FROM nginx:latest
COPY docker/proxy.conf /etc/nginx/
COPY docker/nginx.conf /etc/nginx/
COPY docker/default.conf /etc/nginx/conf.d/
COPY --from=arsenal-ui-builder /opt/arsenal/build /usr/share/nginx/html
CMD ["nginx", "-g", "daemon off;"]
