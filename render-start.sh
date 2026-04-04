#!/bin/sh
set -e
PORT="${PORT:-80}"
sed "s/__PORT__/${PORT}/g" /etc/nginx/render-template.conf > /etc/nginx/conf.d/default.conf
cd /app
uvicorn app:app --host 127.0.0.1 --port 8000 &
exec nginx -g "daemon off;"
