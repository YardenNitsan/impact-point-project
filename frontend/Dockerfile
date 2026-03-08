FROM node:20

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm install -g @angular/cli@19

EXPOSE 4200

CMD ["ng","serve","--host","0.0.0.0"]