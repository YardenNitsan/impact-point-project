import 'dotenv/config'
import express from "express";
import cors from "cors";
import router from './mongo-handler/routes/mongo.route';
import { errorHandler } from './middlewares/error-handler';
import mongoose from 'mongoose';


const ALLOWED_ORIGIN="http://localhost:4200";
const PORT=3000;
const MONGO_URI = "mongodb://localhost:27017/impact-point";

const app = express();
app.use(cors({
  origin: ALLOWED_ORIGIN,
  methods:["GET", "POST", "DELETE"],

  // to allow json only
  allowHeaders: ["Content-Type"]
}));
app.use(express.json());

mongoose
  .connect(MONGO_URI)
  .then(() => console.log("Mongo connected"))
  .catch((err: Error) => {
    console.error(err);
    process.exit(1);
  });

app.use((req, res, next) => {
  console.log("REQ:", req.method, req.url);
  next();
});


app.use("/api/simulation", router);

app.use(errorHandler);

app.listen(PORT, () => {
  console.log("SERVER IS RUNNING");
});