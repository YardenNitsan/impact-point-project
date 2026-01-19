import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";
import { SimulationInput } from "../models/simulationInput.model";

export const getSimulationDetails = async (
  req: Request<{ id: string }>,
  res: Response
) => {
  try {
    const { id } = req.params;

    const result = await SimulationResult.findById(id);
    if (!result) {
      return res.status(404).json({ message: "Simulation result not found" });
    }

    const input = await SimulationInput.findById(result.simulationInputId);
    if (!input) {
      return res.status(404).json({ message: "Simulation input not found" });
    }

    res.status(200).json({
      createdAt: (result as any).createdAt,
      durationMinutes: result.durationMinutes,
      initialData: input.initialData,
      coordinates: result.coordinates
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ message: "Failed to load simulation details" });
  }
};
