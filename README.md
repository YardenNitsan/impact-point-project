# ImpactPoint

ImpactPoint is a web-based visualization project built with Angular and Cesium,
featuring a Node.js backend for data processing and APIs.

## Requirements
- Node.js >= 18
- Angular CLI (v19)
- npm

## Cesium
This project uses a **public, domain-restricted Cesium Ion access token**
intended for browser usage.  
No private credentials are included in this repository.

## Setup

### 1. Clone the repository
- git clone https://github.com/YardenNitsan/impactpoint.git
- cd impactpoint

### 2. Install Dependecies for node backend
- cd services/main-service
- npm install
- npm build
- npm start

server will run on: http://localhost:3000

### 3. Install Dependecies for python backend
- cd services/ImpactServicePython
- pip install numpy matplotlib fastapi uvicorn

### to run: 
- cd services/ImpactServicePython/services/algorithm_service
- python -m uvicorn main:app --reload

server will run on http://localhost:8000

to send packets easy for demo you can go to http://localhost:8000/docs


### 4. Install dependencies for frontend
- cd frontend
- npm install
- npm install -g @angular/cli@v19
- ng serve
frontend will run on: http://localhost:4200

## Database (MongoDB)

This project uses **MongoDB** as its database, accessed via a dedicated
MongoDB microservice built with **Node.js + Express + Mongoose**.

The database stores:
- **Simulation Inputs** (user-submitted data)
- **Simulation Results** (processed simulation output)

### MongoDB Connection
By default, the MongoDB service connects to:

mongodb://localhost:27017/impact-point

---

## Running MongoDB with Docker (Recommended)

The easiest way to run MongoDB is using Docker.
here, I used docker desktop because I I tested it on windows.

### 1. Run MongoDB container
```bash
docker run -d \
  --name impact-point-mongo \
  -p 27017:27017 \
  mongo
```
mongo will run on port 27017. 
make sure this port is not used




