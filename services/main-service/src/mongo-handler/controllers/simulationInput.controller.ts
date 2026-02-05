import { Request, Response } from "express";
import { SimulationInput } from "../models/simulationInput.model";

export const createSimulation = async (
  req: Request,
  res: Response
) => {
  try {
    const simulation = new SimulationInput({
      initialData: req.body,
    });

    const saved = await simulation.save();
    res.status(201).json(saved);
  } catch {
    res.status(400).json({ error: "Failed to save simulation" });
  }
};
