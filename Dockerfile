# Single image for Render: static UI (nginx) + FastAPI on localhost (same as docker-compose, one container).
FROM node:20-alpine AS frontend-build

WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ARG VITE_API_URL=
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

COPY --from=frontend-build /fe/dist /usr/share/nginx/html
COPY render-nginx.conf.template /etc/nginx/render-template.conf
COPY render-start.sh /render-start.sh
RUN chmod +x /render-start.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 80

CMD ["/render-start.sh"]
