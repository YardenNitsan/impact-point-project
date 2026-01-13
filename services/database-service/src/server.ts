import express from "express";
import mongoose from "mongoose";
import router from "./routes/simulation.routes";

const app = express();
const PORT = 4000;
const MONGO_URI = "mongodb://localhost:27017/impact-point";

app.use(express.json());

// Mongo connection
mongoose
  .connect(MONGO_URI)
  .then(() => console.log("Mongo connected"))
  .catch((err: Error) => {
    console.error(err);
    process.exit(1);
  });

// Routes
app.use("/api/simulation", router);

app.listen(PORT, () => {
  console.log(`Mongo service running on port ${PORT}`);
});