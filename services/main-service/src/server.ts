import "dotenv/config";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import mongoose from "mongoose";
import { errorHandler } from "./middlewares/error-handler";
import { environmentService } from "./environment";
import router from "./mongo-handler/routes/mongo.route";
import { globalRateLimiter, perIpRateLimiter } from "./middlewares/rate-limit";

const ALLOWED_ORIGIN = environmentService.ALLOWED_ORIGIN;
const PORT = environmentService.NODE_PORT;
const MONGO_URI = environmentService.MONGO_URI;

const app = express();

app.disable("x-powered-by");
mongoose.set("sanitizeFilter", true);

// If you deploy behind nginx / traefik / cloudflare, enable this correctly.
// app.set("trust proxy", 1);

app.use(helmet());

app.use(
  cors({
    origin: ALLOWED_ORIGIN,
    methods: ["GET", "POST", "DELETE"],
    allowedHeaders: ["Content-Type"],
    optionsSuccessStatus: 204,
  }),
);

app.use(express.json({ limit: "1mb", strict: true, type: "application/json" }));

app.use(globalRateLimiter);
app.use(perIpRateLimiter);

app.use((req, res, next) => {
  const startedAt = Date.now();

  res.on("finish", () => {
    console.log(
      `${req.ip} ${req.method} ${req.originalUrl} -> ${res.statusCode} ${Date.now() - startedAt}ms`,
    );
  });

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
  .connect(MONGO_URI, {
    serverSelectionTimeoutMS: 5000,
  })
  .then(() => {
    console.log("Mongo connected");
    app.listen(PORT, () => {
      console.log(`SERVER IS RUNNING ON PORT ${PORT}`);
    });
  })
  .catch((err: Error) => {
    console.error("Mongo connection error:", err.message);
    process.exit(1);
  });
