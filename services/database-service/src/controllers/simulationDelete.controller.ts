import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";
import { SimulationInput } from "../models/simulationInput.model";

export const deleteSimulation = async (
  req: Request<{ id: string }>,
  res: Response
) => {
  try {
    const result = await SimulationResult.findById(req.params.id);
    if (!result) {
      return res.status(404).json({ error: "sim not found" });
    }

    await SimulationResult.findByIdAndDelete(req.params.id);
    await SimulationInput.findByIdAndDelete(result.simulationInputId);

    res.status(200).json({ message: "simulation deleted successfully" });
  } catch {
    res.status(500).json({ error: "Failed to delete simulation" });
  }
};
