import 'dotenv/config'
import express from "express";
import cors from "cors";
import coordsRoute from "./routes/coords-routes";

const app = express();
app.use(cors({
  origin: "http://localhost:4200",
  methods:["GET", "POST"],
  allowHeaders: ["Content-Type"]
}));
app.use(express.json());

app.use("/api/coords", coordsRoute);

app.listen(3000, () => {
  console.log("SERVER IS RUNNING");
});
