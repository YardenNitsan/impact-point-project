import { Request, Response } from "express";
import { SimulationResult } from "../models/simulationResult.model";

export const deleteSimulation = async (
  req: Request<{ id: string }>,
  res: Response
) => {
  try {
    const { id } = req.params;

    const deleted = await SimulationResult.findByIdAndDelete(id);

    if (!deleted) {
      return res.status(404).json({ error: "Simulation not found" });
    }

    res.status(200).json({ message: "Simulation deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: "Failed to delete simulation" });
  }
};
