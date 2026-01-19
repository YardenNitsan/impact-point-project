import { Router, Request, Response } from "express";
import { saveSimulation, getSimulations, deleteSimulation, getSimulationToWatch } from "../services/mongo.service";
import { getSimulationDetails } from "../services/mongo.service";


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

routerMongo.get("/:id", async (req, res) => {
  try{
    const {id} = req.params;
    console.log(id);
    const simulation = await getSimulationToWatch(id);
    res.status(200).json(simulation);
  } catch{
    res.status(500).json({message: 'Mongo service unavailable'});
  }
})

routerMongo.get("/:id/details", async (req, res) => {
  try {
    const { id } = req.params;
    const details = await getSimulationDetails(id);
    res.status(200).json(details);
  } catch {
    res.status(500).json({ message: "Mongo service unavailable" });
  }
});


routerMongo.delete("/:id", async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    await deleteSimulation(id);
    res.sendStatus(204);
  } catch {
    res.status(500).json({ error: "Mongo service unavailable" });
  }
});

export default routerMongo;