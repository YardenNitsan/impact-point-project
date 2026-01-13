import 'dotenv/config'
import express from "express";
import cors from "cors";
import routerMongo from './routes/mongo.route';
import { errorHandler } from './middlewares/error-handler';


const ALLOWED_ORIGIN="http://localhost:4200";
const PORT=3000;

const app = express();
app.use(cors({
  origin: ALLOWED_ORIGIN,
  methods:["GET", "POST"],

  //to allow json only
  allowHeaders: ["Content-Type"]
}));
app.use(express.json());

app.use("/api/simulation", routerMongo);

app.use(errorHandler);

app.listen(PORT, () => {
  console.log("SERVER IS RUNNING");
});


