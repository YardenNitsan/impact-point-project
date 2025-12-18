import 'dotenv/config'
import express from "express";
import cors from "cors";
import coordsRoute from "./routes/coords";

const app = express();
app.use(cors());
app.use(express.json());

app.use("/api/coords", coordsRoute);
const CESIUM_TOKEN = process.env.CESIUM_TOKEN;
console.log(CESIUM_TOKEN);

app.listen(3000, () => {
  console.log("SERVER IS RUNNING");
});
