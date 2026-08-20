# 多阶段构建：前端 Vite 构建 → nginx 托管静态 + 反代后端
# 服务器无需预装 Node，`docker compose up --build` 自动完成前端构建

FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
