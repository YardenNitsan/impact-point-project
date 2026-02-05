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
      return res.status(404).json({ error: "Simulation not found" });
    }

    // delete result
    await SimulationResult.deleteOne({ _id: result._id });

    // delete input if exists
    if (result.simulationInputId) {
      await SimulationInput.deleteOne({ _id: result.simulationInputId });
    }

    res.status(200).json({ message: "Simulation deleted" });
  } catch (err) {
    console.error("DELETE ERROR:", err);
    res.status(500).json({ error: "Delete failed" });
  }
};
