import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";

export const getSimulationResults = async (
  _req: Request,
  res: Response
) => {
  const results = await SimulationResult.find()
    .populate("simulationInputId")
    .sort({ createdAt: -1 });

  res.json(results);
};
