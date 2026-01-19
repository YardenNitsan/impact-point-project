import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";

export const getSimulationResults = async (
  _req: Request,
  res: Response
) => {
  try {
    const results = await SimulationResult.find()
      .sort({ createdAt: -1 });

    const response = results.map(r => ({
      id: r._id,
      durationMinutes: r.durationMinutes,
      createdAt: (r as any).createdAt
    }));

    res.status(200).json(response);
  } catch {
    res.status(500).json({ message: "could not return results" });
  }
};
