# ImpactPoint

### Physics‑Driven Trajectory Simulation & 3D Visualization Platform

ImpactPoint is an advanced simulation platform designed to model,
analyze, and visualize projectile trajectories using a combination of
**physics-based numerical simulation**, **machine learning
approximation**, and **interactive 3D geospatial rendering**.

The system combines multiple engineering domains:

- Computational physics
- Numerical simulation
- Machine learning approximation
- Distributed system architecture
- 3D geospatial visualization

The goal of the platform is to allow users to launch simulated
trajectories from any point on Earth and observe the full flight path
and impact point in a **real-time interactive 3D environment**.

The system integrates:

- **3DOF ballistic physics engine**
- **Machine learning trajectory approximation model (KNN)**
- **Interactive CesiumJS WebGL globe**
- **Service-oriented backend architecture**

This project is designed as a **modular microservice system**, where
each component performs a clearly defined role such as orchestration,
simulation, visualization, or prediction.

---

# System Overview

ImpactPoint is composed of several independent services that communicate
through REST APIs.

Key services include:

- **Frontend Visualization**
- **API Orchestration Server**
- **Physics Simulation Engine**
- **Machine Learning Prediction Service**
- **Simulation and Dataset Databases**

Each service runs independently and can be scaled, deployed, or tested
in isolation.

---

# Core Capabilities

## Physics-Based Trajectory Simulation

At the heart of ImpactPoint is a **3 Degree of Freedom (3DOF) ballistic
solver** implemented in Python.

The simulation engine models projectile motion using numerical
integration techniques and physical forces.

Capabilities include:

- Gravity-based motion modeling
- Aerodynamic drag computation
- Atmospheric condition modeling
- Wind influence along the trajectory
- Terrain-aware impact detection
- Numerical timestep integration

The solver produces trajectories consisting of **thousands of sampled
spatial positions**, which are later used for visualization and machine
learning dataset generation.

---

## Machine Learning Trajectory Approximation

In addition to the physics simulation engine, ImpactPoint includes a
**machine learning model** designed to approximate projectile
trajectories.

The ML system uses a **K-Nearest Neighbors (KNN)** approach to estimate
trajectories based on previously simulated data.

Pipeline:

1.  Randomized launch parameters are generated.
2.  Full physics simulations are executed.
3.  Trajectories are normalized.
4.  Motion is encoded as relative displacement sequences.
5.  KNN identifies the closest trajectories in the dataset.
6.  A weighted average produces the predicted trajectory.

The ML model is designed to provide **fast approximations of
computationally expensive simulations**.

---

## Interactive 3D Visualization

The frontend uses **Angular and CesiumJS** to render a fully interactive
3D globe.

Users can visualize:

- Complete projectile trajectories
- Real-time animated flight paths
- Terrain-aware impact points
- Camera adaptive rendering
- Trajectory interpolation
- High-performance rendering of simulation data

The visualization layer allows the system to display thousands of
trajectory points efficiently within a WebGL environment.

---

# Architecture

ImpactPoint uses a **service orchestration architecture** designed for
modularity and scalability.

Frontend → API Server → Physics Engine → ML Service → Databases

Each service is responsible for its own logic and communicates through
REST endpoints.

---

# Project Structure

The repository is organized into multiple top-level components.

    impactpoint
    │
    ├── frontend
    │   Angular + Cesium visualization layer
    │
    ├── services
    │   ├── main-service        # Node.js API orchestration server
    │   ├── physics-service     # Python ballistic simulation engine
    │   └── ml-model-service    # Machine learning dataset & prediction service
    │
    ├── database
    │
    └── docs

---

# Requirements

The project has been tested with the following environment:

    Node.js v24.11.1
    Python v3.13.12
    Angular CLI v19
    MongoDB
    Docker (optional but recommended)

---

# Setup

## Clone Repository

    git clone https://github.com/YardenNitsan/impactpoint.git
    cd impactpoint

---

# Original Detailed Setup Instructions

The following sections preserve the original setup and run instructions
provided with the project.

These steps describe how to run each component independently during
development.

---

# ImpactPoint

ImpactPoint is a physics‑driven trajectory simulation and visualization
platform.

The system integrates:

- a **3DOF ballistic physics engine**
- a **machine learning trajectory approximation model (KNN)**
- an **interactive 3D geospatial visualization interface** built with
  **Angular + CesiumJS**

Users can launch simulated trajectories from any geographic coordinate
and visualize the full flight path and impact point directly on a **3D
globe**.

The platform is built using a **service‑oriented architecture**
separating:

- visualization
- API orchestration
- physics simulation
- machine learning prediction

Each service manages its own responsibilities and storage.

---

# Key Features

## Physics Simulation Engine

The core of ImpactPoint is a **3DOF ballistic trajectory solver**
implemented in Python.

Capabilities:

- gravity‑based projectile dynamics
- aerodynamic drag modeling
- atmospheric modeling
- wind influence along trajectory
- terrain‑aware impact detection
- numerical integration using fixed timestep solvers

The solver generates full trajectories consisting of **thousands of
sampled positions**.

---

## Machine Learning Trajectory Approximation

ImpactPoint includes a **K‑Nearest Neighbors (KNN)** model for fast
trajectory prediction.

Pipeline:

1.  Random launch parameters are generated
2.  Full physics simulations are executed
3.  Trajectories are normalized
4.  Motion represented as relative displacement sequences
5.  KNN finds nearest trajectories
6.  Weighted averaging produces predicted trajectory

The ML service maintains its own **dataset database**.

---

## 3D Visualization

The frontend provides an interactive **CesiumJS WebGL globe**.

Features:

- real‑time trajectory rendering
- animated projectile flight
- terrain‑aware impact visualization
- camera adaptive rendering
- trajectory interpolation
- dynamic sampling of simulation data

---

# System Architecture

ImpactPoint uses a **service orchestration architecture**.

                     +----------------------+
                     |      Frontend        |
                     |   Angular + Cesium   |
                     +----------+-----------+
                                |
                                | REST API
                                |
                     +----------v-----------+
                     |      API Server      |
                     |    Node.js / Express |
                     +----------+-----------+
                                |
                                | Simulation Request
                                |
                     +----------v-----------+
                     |    Physics Engine    |
                     |    Python + FastAPI  |
                     |     3DOF Solver      |
                     +----------+-----------+
                                |
                                | Simulation Result
                                |
                     +----------v-----------+
                     |   ML Model Service   |
                     |    KNN Prediction    |
                     +----------+-----------+
                                |
                    +-----------+-----------+
                    |                       |
            +-------v--------+     +--------v--------+
            | Simulation DB  |     |   Dataset DB    |
            |  (API Server)  |     |  (ML Service)   |
            +----------------+     +-----------------+

### Service Responsibilities

Service Technology Responsibility

---

Frontend Angular + Cesium Visualization
API Server Node.js / Express Simulation orchestration
Physics Engine Python / FastAPI 3DOF trajectory simulation
ML Model Service Python KNN trajectory approximation

---

# Project Structure

    impactpoint
    │
    ├── frontend
    │   Angular + Cesium visualization
    │
    ├── services
    │   ├── main-service        # Node.js API
    │   ├── physics-service     # Python simulation engine
    │   └── ml-model-service    # KNN dataset & prediction
    │
    ├── database
    │
    └── docs

---

# Requirements

Tested with:

    Node.js v24.11.1
    Python v3.13.12
    Angular CLI v19
    MongoDB
    Docker (optional)

---

# Setup

## Clone Repository

    git clone https://github.com/YardenNitsan/impactpoint.git
    cd impactpoint

---

# API Server (Node.js + TypeScript)

The API server is implemented using **Node.js, Express, and
TypeScript**.

Install dependencies:

    cd services/main-service
    npm install
    npm install express cors axios dotenv mongoose

# Running the API Server

The API supports two modes of execution.

## Development Mode (Recommended for Development)

Used during active development.

The server runs using **nodemon + ts-node**, which means:

- TypeScript is executed directly without compiling.
- The server automatically restarts when source files change.
- No manual restart is required.

Install dependencies for development:

    npm install -D typescript ts-node nodemon @types/node @types/express

Create tsconfig.json if you don't have:

    npx tsc --init

Example tsconfig.json:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "moduleResolution": "node",
    "rootDir": "./src",
    "outDir": "./dist",
    "esModuleInterop": true,
    "strict": false,
    "skipLibCheck": true
  },
  "include": ["src"]
}
```

Add scripts to package.json:

```json
"scripts": {
  "dev": "nodemon --watch src --exec ts-node src/server.ts",
  "build": "tsc",
  "start": "node dist/server.js"
}
```

Start the server:

    npm run dev

Server will run at:

    http://localhost:3000

---

## Production Mode

Step 1 --- Compile TypeScript:

    npm run build

Step 2 --- Start compiled server:

    npm start

---

# Start Physics Service

Install dependencies:

    cd services/algorithm_service
    pip install numpy matplotlib fastapi uvicorn requests

Run service:

    cd services/algorithm_service
    python -m uvicorn main:app --reload

Server:

    http://localhost:8000

API Documentation:

    http://localhost:8000/docs

---

# Start Frontend

    cd frontend
    npm install
    npm install -g @angular/cli@19
    ng serve

Frontend runs at:

    http://localhost:4200

---

# Development Orchestration Script

To simplify local development, the project includes a **development
orchestration script**:

    dev.ps1

This script automatically starts and stops the entire system in the
correct order.

Services started:

- MongoDB container (Docker)
- Physics Engine service (Python / FastAPI)
- API Server (Node.js / Express)
- Frontend application (Angular)

Each service is started only after the previous one becomes available.

The script stores process identifiers in:

    .dev/dev_pids.json

allowing all services to be stopped safely.

## Running the Entire Project

From the **project root folder**:

Start everything:

```powershell
powershell ./dev.ps1 start
```

Stop everything:

```powershell
powershell ./dev.ps1 stop
```

Restart the environment:

```powershell
powershell ./dev.ps1 restart
```

Check system status:

```powershell
powershell ./dev.ps1 status
```

This script ensures the correct startup order:

MongoDB → Physics Engine → API Server → Frontend

---

# Databases

ImpactPoint uses two independent data stores.

### Simulation Database

Managed by the **API server**.

Stores:

- simulation inputs
- simulation results
- trajectory metadata

# Running MongoDB with Docker

If you don't have MongoDB installed locally, you can run it using
Docker.

Start a MongoDB container:

```bash
docker run -d \
  --name impactpoint-mongo \
  -p 27017:27017 \
  mongo
```

### Dataset Database

Managed by the **ML service**.

Stores:

- trajectory dataset
- normalized trajectories
- ML training data

---

# Docker Deployment (Recommended)

ImpactPoint can run entirely using **Docker Compose**, allowing the
entire system to start with a single command and without installing
Node, Python, or MongoDB locally.

Docker runs the following services as isolated containers:

- Frontend (Angular + Cesium)
- API Server (Node.js + Express)
- Physics Engine (Python + FastAPI)
- MongoDB database

All containers communicate through an internal Docker network.

## Architecture (Docker)

Browser → Frontend Container → API Container → Physics Container →
MongoDB

Container communication:

- API Server (Main Service) communicates with the Physics Engine
  service: http://algorithm:8000

- API Server (Main Service) communicates with the MongoDB database:
  mongodb://mongo:27017

No other services communicate directly with the database or with each
other. All requests are orchestrated through the API Server.

These hostnames work because Docker provides **automatic DNS
resolution** between containers inside the same compose network.

## Running the Entire System with Docker

From the **project root directory**:

Build and start all services:

    docker compose up --build

note: this will start the algorithm service with only one worker. to
build the algorithm with multiple workers change the number in the
docker-compose.yml to a desired number and run with:

    docker compose --profile dataset up

Docker will automatically start:

- MongoDB container
- Physics simulation service
- Node.js API server
- Angular frontend

After startup the system will be available at:

Frontend:

    http://localhost:4200

API Server:

    http://localhost:3000

Physics Engine API:

    http://localhost:8000/docs

## Stopping the System

Stop containers:

    CTRL + C

Then cleanly remove containers and networks:

    docker compose down

## Restarting

If containers were already built:

    docker compose up

If Dockerfiles or dependencies changed:

    docker compose up --build

## Viewing Logs

View logs from all services:

    docker compose logs -f

View logs for a specific service:

    docker compose logs main
    docker compose logs algorithm
    docker compose logs frontend
    docker compose logs mongo

## Verifying Containers

List running containers:

    docker ps

Expected containers:

    impact-point-project-main-1
    impact-point-project-frontend-1
    algorithm-service
    impactpoint-mongo

This confirms the full **microservice architecture is running inside
Docker**.

---

# Technologies

### Frontend

- Angular
- CesiumJS
- TypeScript

### Backend

- Node.js
- Express

### Physics Engine

- Python
- FastAPI
- NumPy

### Machine Learning

- Python
- KNN

### Database

- MongoDB
- Mongoose

---

# License

Educational / research project.
