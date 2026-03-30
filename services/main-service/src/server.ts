import "dotenv/config";
import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import { errorHandler } from "./middlewares/error-handler";
import { environmentService } from "./environment";
import router from "./mongo-handler/routes/mongo.route";

const ALLOWED_ORIGIN = environmentService.ALLOWED_ORIGIN;
const PORT = environmentService.NODE_PORT;
const MONGO_URI = environmentService.MONGO_URI;

const app = express();

app.use(
  cors({
    origin: ALLOWED_ORIGIN,
    methods: ["GET", "POST", "DELETE"],
    allowedHeaders: ["Content-Type"],
  }),
);

app.use(express.json({ limit: "1mb" }));

app.use((req, res, next) => {
  console.log("REQ:", req.method, req.url);
  next();
});

app.get("/health", (_req, res) => {
  res.json({
    success: true,
    service: "node-gateway",
    message: "Node gateway is running",
  });
});

app.use("/api/simulation", router);

app.use(errorHandler);

mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log("Mongo connected");
    app.listen(PORT, () => {
      console.log(`SERVER IS RUNNING ON PORT ${PORT}`);
    });
  })
  .catch((err: Error) => {
    console.error("Mongo connection error:", err);
    process.exit(1);
  });
