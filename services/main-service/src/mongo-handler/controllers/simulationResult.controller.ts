import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";

export const getSimulationResults = async (_req: Request, res: Response) => {
  try {
    const results = await SimulationResult.find().sort({ createdAt: -1 });

    res.status(200).json(
      results.map((r) => ({
        id: r._id.toString(),
        createdAt: r.createdAt,
        durationSeconds: r.durationSeconds,
        weather_source: r.weather_source,
      })),
    );
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Failed to fetch simulations" });
  }
};
