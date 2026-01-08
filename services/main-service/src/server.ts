import 'dotenv/config'
import express from "express";
import cors from "cors";
import routerCoords from "./coords/coords.route";
import routerTrajectory from './trajectory/trajectory.route';
import { errorHandler } from './middlewares/error-handler';


const app = express();
app.use(cors({
  origin: "http://localhost:4200",
  methods:["GET", "POST"],

  //to allow json only
  allowHeaders: ["Content-Type"]
}));
app.use(express.json());

app.use("/api/coords", routerCoords);

app.use("/api/trajectory", routerTrajectory);

app.use(errorHandler);

app.listen(3000, () => {
  console.log("SERVER IS RUNNING");
});
