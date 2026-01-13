import { Router, Request, Response } from "express";
import { saveSimulation, getSimulations } from "../services/mongo.service";

const routerMongo = Router();

routerMongo.post("/", async (req: Request, res: Response) => {
  try {
    const result = await saveSimulation(req.body);
    res.status(201).json(result);
  } catch {
    res.status(500).json({ error: "Mongo service unavailable" });
  }
});

routerMongo.get("/", async (_req: Request, res: Response) => {
  try {
    const result = await getSimulations();
    res.json(result);
  } catch {
    res.status(500).json({ error: "Mongo service unavailable" });
  }
});

export default routerMongo;
