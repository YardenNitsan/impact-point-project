import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";

export const getSimulationToWatch = async (
  req: Request,
  res: Response
) => {
  try {
    const sim = await SimulationResult.findById(req.params.id);

    if (!sim) {
      return res.status(404).json({ error: "Simulation not found" });
    }

    // 🔥 הפרונט מצפה Coordinate[]
    res.status(200).json(sim.coordinates);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to load simulation" });
  }
};
