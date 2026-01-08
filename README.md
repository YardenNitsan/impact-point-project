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

### 2. Install Dependecies for backend
- cd services/main-service
- npm install
- npm build
- npm start

server will run on: http://localhost:3000

### 3. Install dependencies for frontend
- cd frontend
- npm install
- ng serve

frontend will run on: http://localhost:4200


