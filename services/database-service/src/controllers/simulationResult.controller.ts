import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";

interface CreateSimulationResultBody {
  simulationInputId: string;
  coordinates: { lon: number; lat: number; alt: number }[];
  durationMinutes: number;
}

export const createSimulationResult = async (
  req: Request<{}, {}, CreateSimulationResultBody>,
  res: Response
) => {
  try {
    const result = new SimulationResult({
      simulationInputId: req.body.simulationInputId,
      coordinates: req.body.coordinates,
      durationMinutes: req.body.durationMinutes,
    });

    await result.save();
    res.sendStatus(201);
  } catch {
    res.status(400).json({ error: "Failed to save simulation result" });
  }
};

export const getSimulationResults = async (
  _req: Request,
  res: Response
) => {
  const results = await SimulationResult.find()
    .populate("simulationInputId")
    .sort({ createdAt: -1 });

  res.json(results);
};
