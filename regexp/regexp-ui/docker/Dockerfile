FROM nginx:1.12.2-alpine
COPY build /usr/share/nginx/html
COPY docker/proxy.conf /etc/nginx/
COPY docker/nginx.conf /etc/nginx/
COPY docker/default.conf /etc/nginx/conf.d/
