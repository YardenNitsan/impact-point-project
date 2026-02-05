import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";
import { SimulationInput } from "../models/simulationInput.model";

export const getSimulationDetails = async (
  req: Request,
  res: Response
) => {
  try {
    const result = await SimulationResult.findById(req.params.id);

    if (!result) {
      return res.status(404).json({ error: "Simulation not found" });
    }

    const input = await SimulationInput.findById(
      result.simulationInputId
    );

    if (!input) {
      return res.status(404).json({ error: "Simulation input not found" });
    }

    res.status(200).json({
      createdAt: result.createdAt,
      durationMinutes: result.durationMinutes,
      initialData: input.initialData,
      coordinates: result.coordinates,
    });
  } catch (err) {
    console.error("DETAILS ERROR:", err);
    res.status(500).json({ error: "Failed to load simulation details" });
  }
};
